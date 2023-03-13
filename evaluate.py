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
    B = random.randint(0,batch_size)
    for batchs in loader:
        num = 0
        A = random.randint(0,len(batchs))
        for batch in batchs:
            with torch.no_grad():
                if sys_one & (num == A):
                    batch2 = (batch[0][B], batch[1][B],np.array(batch[2][B]), batch[3][B], np.array(batch[4][B]), batch[5])
                    batch2 = to_device(batch2, device)
                    batch2 = fff(batch2)
                    output2 = model(*(batch2), n_timesteps=50, temperature=1, stoc=False, length_scale=1)
                    batch22 = to_device(batch, device)
                    sys_one = False
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])
            num += 1
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    message = "Validation Step {}, Total Loss: {:.4f}, Dur Loss: {:.4f}, Prior PostNet Loss: {:.4f}, " \
              "Diff Loss: {:.4f}".format(*[step] + list(loss_means))

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            B,
            batch22,
            output2,
            vocoder,
            model_config,
            preprocess_config,
            save_path
        )
        write(os.path.join(save_path,"{}_wav_reconstruction.wav").format(batch22[0][0][:-4]), 22050, wav_reconstruction)
        write(os.path.join(save_path,"{}_wav_prediction.wav").format(batch22[0][0][:-4]), 22050, wav_prediction)

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=np.squeeze(wav_reconstruction),
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=np.squeeze(wav_prediction),
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


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

    # Get model
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    os.makedirs('./Temp/logs',exist_ok = True)
    logger = SummaryWriter('./Temp/logs')
    save_path = './Temp'
    message = evaluate(save_path, model, args.restore_epoch, configs, logger, vocoder)
    print(message)
