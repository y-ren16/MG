import hifigan
import os
import json
import torch
from model.gradtts import GradTTS
import numpy as np


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = GradTTS(preprocess_config, model_config).to(device)

    if args.restore_epoch:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pt".format(args.restore_epoch),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        model.train()
        learning_rate = train_config["optimizer"]["learning_rate"]
        scheduled_optim = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate
        )
        return model
        # , scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "HiFi-GAN":
        HIFIGAN_CONFIG = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/config.json'
        HIFIGAN_CHECKPT = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/g_02500000'
        with open(HIFIGAN_CONFIG, "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(HIFIGAN_CHECKPT)
        elif speaker == "universal":
            ckpt = torch.load(HIFIGAN_CHECKPT)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
