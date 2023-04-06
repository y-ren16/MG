import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from utils.mymodel import get_model, get_vocoder
from utils.tools import to_device, log
from model import GradLoss
from dataset import TextMelDataset
from utils.syn import synth_one_sample
import json
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write 
import numpy as np
from model import GradTTS
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(save_path, model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = TextMelDataset(
        "valid.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn_DDP,
    )

    # Get loss function
    Loss = GradLoss(preprocess_config, model_config).to(device)

    # generator = GradTTS(preprocess_config, model_config).to(device)
    # checkpoint = torch.load(
    #         os.path.join(
    #             train_config["path"]["ckpt_path"], preprocess_config["dataset"], "2023-03-13-03_54", f'80.pt'
    #         ) , map_location=lambda loc, storage: loc
    #     )
    # generator.load_state_dict(checkpoint)
    # _ = generator.eval()
    # print(model==generator)

    # Evaluation
    # 4 Total;
    def fff(bb):
        return (bb[2].unsqueeze(0), bb[3].unsqueeze(0), bb[4].unsqueeze(0), bb[5])
    loss_sums = [0 for _ in range(4)]
    import random
    sys_one = True
    num_batches = 0
    for batch in loader:
        num_batches += 1
    # A = random.randint(0,num_batches-2)
    # B = random.randint(0,batch_size-1)
    # B = 0
    num = 0
    for iii, batch in tqdm(enumerate(loader)):
        with torch.no_grad():
            for B in range(batch_size):
                batch2 = (batch[0][B], batch[1][B],np.array(batch[2][B]), batch[3][B], np.array(batch[4][B]), batch[5])
                batch2 = to_device(batch2, device)
                batch2 = fff(batch2)
                output2 = model(*(batch2))
                batch22 = to_device(batch, device)
                fig, fig_prediction, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                    B,
                    batch22,
                    output2,
                    vocoder,
                    model_config,
                    preprocess_config,
                    save_path,
                    step
                )
                spec_path = os.path.join(save_path, batch[0][B][:-3] + '.npy')
                np.save(spec_path, wav_prediction)
                num += 1

    return num


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--restore_epoch", 
        type=int, 
        default=792
        )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default="config/NEB/preprocess.yaml",
        # required=True,
        help="path to preprocess.yaml"
    )
    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        default="config/NEB/model.yaml",
        # required=True, 
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", 
        "--train_config", 
        type=str, 
        default="config/NEB/train.yaml",
        # required=True, 
        help="path to train.yaml"
    )
    parser.add_argument( 
        "--time_dir", 
        type=str, 
        default="2023-03-29-00_01",
        # required=True, 
        help="path to ckpt"
    )
    args = parser.parse_args()
    # with open('./eva_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    if args.restore_epoch is None:
        ii = []
        for pt_list in os.listdir(os.path.join(train_config["path"]["ckpt_path"],preprocess_config["dataset"],args.time_dir)):
            ii.append(int(pt_list.split('.')[0]))
        args.restore_epoch=max(ii)

    # Get model
    model = get_model(args, configs, device, args.time_dir, train=False)
    vocoder = get_vocoder(model_config, device)
    save_path = os.path.join('../Temp_npy', preprocess_config["dataset"], str(args.restore_epoch))
    log_path = os.path.join(save_path, 'logs')
    os.makedirs(log_path,exist_ok = True)
    logger = SummaryWriter(log_path)
    message = evaluate(save_path, model, args.restore_epoch, configs, logger, vocoder)
    print(message)
