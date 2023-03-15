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
from utils.mymodel import vocoder_infer

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
        collate_fn=dataset.collate_fn,
    )
    def fff(bb):
        return (bb[2].unsqueeze(0), bb[3].unsqueeze(0), bb[4].unsqueeze(0), bb[5])
    num = 0
    for batchs in tqdm(loader):
        for batch in batchs:
            with torch.no_grad():
                for B in tqdm(range(len(batch[0]))):
                    batch2 = (batch[0][B], batch[1][B],np.array(batch[2][B]), batch[3][B], np.array(batch[4][B]), batch[5])
                    batch2 = to_device(batch2, device)
                    batch2 = fff(batch2)
                    output2 = model(*(batch2), n_timesteps=50, temperature=1, stoc=False, length_scale=1)

                    wav_prediction = vocoder_infer(
                        output2[8],
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    write(os.path.join(save_path,"step_{}_{}_wav_prediction.wav").format(step, batch[0][B][:-4]), 22050, wav_prediction)
                num += 1
    print(num)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_epoch", type=int, default=30000)
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
    args = parser.parse_args()
    with open('./eva_args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    args.restore_epoch = None
    if args.restore_epoch is None:
        ii = [
            int(pt_list.split('.')[0])
            for pt_list in os.listdir(
                os.path.join(
                    train_config["path"]["ckpt_path"],
                    preprocess_config["dataset"],
                    train_config["path"]["time"],
                )
            )
        ]
        args.restore_epoch=max(ii)

    # Get model
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    os.makedirs('./fake1',exist_ok = True)
    save_path = './fake1'
    evaluate(save_path, model, args.restore_epoch, configs, vocoder=vocoder)

