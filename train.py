import argparse
import yaml
from tqdm import tqdm
import json
import numpy as np

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import TextMelDataset
from utils.mymodel import get_model, get_param_num, get_vocoder
from model.loss import GradLoss
from utils.tools import to_device, log
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# from evaluate import evaluate

gpus = [0, 1, 2, 4, 5, 6, 7]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")

def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs

    torch.manual_seed(train_config["random_seed"])
    np.random.seed(train_config["random_seed"])

    dataset_name = preprocess_config["dataset"]

    dataset = TextMelDataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = train_config["optimizer"]["group_size"] # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    print(f"Batch size: {batch_size}, Group size: {group_size}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    model = get_model(args, configs, device, train=True)

    # model = nn.DataParallel(model)
    model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
    learning_rate = train_config["optimizer"]["learning_rate"]
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate
    )

    num_param = get_param_num(model)

    Loss = GradLoss(preprocess_config, model_config).to(device)
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

    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    out_size = model_config["out_size"]

    outer_bar = tqdm(total=total_epoch, desc="Training", position=0)
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                # Forward
                # out_size = torch.LongTensor(out_size)
                # output = model(*(batch[2:]), out_size=out_size)
                output = model(*(batch[2:]))

                losses = Loss(batch, output)
                total_loss = losses[0]

                total_loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad()

                losses = [l.item() for l in losses]
                log(train_logger, step, losses=losses)
                step += 1

            inner_bar.update(1)

        if step % log_epoch == 0:
            message1 = "Epoch {}/{}, Step{}, ".format(epoch, total_epoch, step)
            message2 = "Total Loss: {:.4f}, Dur Loss: {:.4f}, Prior PostNet Loss: {:.4f}, Diff Loss: {:.4f}" \
                .format(*losses)
            with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                f.write(message1 + message2 + "\n")
            outer_bar.write(message1 + message2)

        # if step % val_epoch == 0:
        #     model.eval()
        #     message = evaluate(model, step, configs, val_logger, vocoder)
        #     with open(os.path.join(val_log_path, "log.txt"), "a") as f:
        #         f.write(message + "\n")
        #     outer_bar.write(message)
        #
        #     model.train()

        if step % save_epoch == 0:
            ckpt = model.state_dict()
            torch.save(ckpt, os.path.join(ckpt_path, "{}.pt".format(epoch)))
            # torch.save(
            #     {
            #         "model": model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #     },
            #     os.path.join(
            #         ckpt_path,
            #         "{}.pt".format(epoch),
            #     ),
            # )

        if epoch == total_epoch:
            quit()
        epoch += 1
        outer_bar.update(1)





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
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open('./commandline_args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
