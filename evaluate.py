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
    A = random.randint(0,num_batches-2)
    B = random.randint(0,batch_size-1)
    # B = 0
    num = 0
    for batch in loader:
        with torch.no_grad():
            if sys_one & (num == A):
                batch2 = (batch[0][B], batch[1][B],np.array(batch[2][B]), batch[3][B], np.array(batch[4][B]), batch[5])
                batch2 = to_device(batch2, device)
                batch2 = fff(batch2)
                output2 = model(*(batch2))
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
        write(os.path.join(save_path,"step_{}_{}_re.wav").format(step, batch22[0][B][:-4]), 22050, wav_reconstruction)
        write(os.path.join(save_path,"step_{}_{}_pre.wav").format(step, batch22[0][B][:-4]), 22050, wav_prediction)

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_gt_{}".format(step, tag),
        )
        log(
            logger,
            fig=fig_prediction,
            tag="Validation/step_{}_pre_{}".format(step, tag),
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
    parser.add_argument("--restore_epoch", type=int, default=None)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        default="config/LJSpeech/preprocess.yaml",
        # required=True,
        help="path to preprocess.yaml"
    )
    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        default="config/LJSpeech/model.yaml",
        # required=True, 
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", 
        "--train_config", 
        type=str, 
        default="config/LJSpeech/train.yaml",
        # required=True, 
        help="path to train.yaml"
    )
    parser.add_argument( 
        "--time_dir", 
        type=str, 
        default="2023-03-25-23_50",
        required=True, 
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
    save_path = os.path.join('../Temp_Audio', preprocess_config["dataset"], str(args.restore_epoch))
    log_path = os.path.join(save_path, 'logs')
    os.makedirs(log_path,exist_ok = True)
    logger = SummaryWriter(log_path)
    message = evaluate(save_path, model, args.restore_epoch, configs, logger, vocoder)
    print(message)
