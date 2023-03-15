import argparse
import yaml
from tqdm import tqdm
import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6, 7"
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import TextMelDataset
from utils.mymodel import get_model, get_param_num, get_vocoder
from model.loss import GradLoss
from utils.tools import to_device, log, pad_1D, pad_2D
import time
from model import GradTTS
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from evaluate import evaluate
# gpus = [0, 1, 2, 4, 5, 6, 7]
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))
# device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
def main(rank, args, configs):
    if rank == 0:
        print("Prepare training ...")
    preprocess_config, model_config, train_config = configs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]['dist_backend'],
            init_method=train_config["dist_config"]['dist_url'],
            world_size=train_config["dist_config"]['world_size'] * num_gpus,
            rank=rank
        )
    torch.cuda.manual_seed(train_config["random_seed"])
    device = torch.device('cuda:{:d}'.format(rank))
    dataset_name = preprocess_config["dataset"]
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = train_config["optimizer"]["group_size"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    dataset = TextMelDataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    train_loader = DataLoader(
        dataset,
        num_workers=group_size,
        shuffle=False,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn
    )
    model = GradTTS(preprocess_config, model_config).to(device)
    if num_gpus > 1:
        model = DDP(model, device_ids=[rank]).to(device)
    learning_rate = train_config["optimizer"]["learning_rate"]
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )
    Loss = GradLoss(preprocess_config, model_config).to(device)
    if rank == 0:
        num_param = get_param_num(model)
        print("Number of Grad-TTS Parameters:", num_param)
        vocoder = get_vocoder(model_config, device)
        # Init logger
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        time_now = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
        train_log_path = os.path.join(train_config["path"]["log_path"], dataset_name, time_now, "train")
        val_log_path = os.path.join(train_config["path"]["log_path"],dataset_name, time_now, "val")
        test_log_path = os.path.join(train_config["path"]["log_path"], dataset_name, time_now, "test")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        os.makedirs(test_log_path, exist_ok=True)
        ckpt_path = os.path.join(train_config["path"]["ckpt_path"], dataset_name, time_now)
        os.makedirs(ckpt_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)
        test_logger = SummaryWriter(test_log_path)
    # Training
    step = args.restore_step + 1
    epoch = args.restore_epoch + 1
    total_step = train_config["step"]["total_step"]
    total_epoch = train_config["step"]["total_epoch"]
    log_step = train_config["step"]["log_step"]
    log_epoch = train_config["step"]["log_epoch"]
    save_step = train_config["step"]["save_step"]
    save_epoch = train_config["step"]["save_epoch"]
    synth_step = train_config["step"]["synth_step"]
    synth_epoch = train_config["step"]["synth_epoch"]
    val_step = train_config["step"]["val_step"]
    val_epoch = train_config["step"]["val_epoch"]
    model.train()
    for epoch in range(total_epoch):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))
        if num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            batch = to_device(batch[0], device, non_blocking=True)
            optimizer.zero_grad()
            output = model(*(batch[2:]))
            losses = Loss(batch, output)
            total_loss = losses[0]
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            optimizer.step()
            if rank == 0:
                losses = [l.item() for l in losses]
                log(train_logger, step, losses=losses)
                print('Steps : {:d}/{:d}, Total Loss: {:.4f}, Dur Loss: {:.4f}, Prior PostNet Loss: {:.4f}, '
                      'Diff Loss: {:.4f}, s/b : {:4.3f}'.format(step,len(train_loader), *losses, time.time() - start_b))
            step += 1
        if rank == 0:
            if epoch % log_epoch == 0:
                message1 = "Epoch {}/{}, Step{}, ".format(epoch, total_epoch, step)
                message2 = "Total Loss: {:.4f}, Dur Loss: {:.4f}, Prior PostNet Loss: {:.4f}, Diff Loss: {:.4f}" \
                    .format(*losses)
                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")
                print(message1)
                print(message2)
            if epoch % save_epoch == 0:
                ckpt = model.module.state_dict()
                torch.save(ckpt, os.path.join(ckpt_path, "{}.pt".format(epoch)))
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
            if epoch % val_epoch == 0:
                evaluate('./Temp', model, epoch, configs, logger=val_logger, vocoder=vocoder)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_step", type=int, default=0)
    # parser.add_argument("--restore_epoch", type=int, default=0)
    # parser.add_argument(
    #     "-p",
    #     "--preprocess_config",
    #     type=str,
    #     required=True,
    #     help="path to preprocess.yaml",
    # )
    # parser.add_argument(
    #     "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    # )
    # parser.add_argument(
    #     "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    # )
    # parser.add_argument('--local_rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open('./commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    # print(args.local_rank)

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # main(args, configs)
    torch.manual_seed(train_config["random_seed"])
    num_gpus = torch.cuda.device_count()

    if train_config["num_gpus"] > 1:
        mp.spawn(main, nprocs=num_gpus, args=(args, configs,))
    else:
        main(0, args, configs)