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
        # model = torch.nn.DataParallel(model)
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"], preprocess_config["dataset"], train_config["path"]["time"], 
            "{}.pt".format(args.restore_epoch),
        )
        ckpt = torch.load(ckpt_path, map_location=lambda loc, storage: loc)
        model.load_state_dict(ckpt)

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


def get_vocoder(con, device):
    name = con["vocoder"]["model"]
    speaker = con["vocoder"]["speaker"]

    if name == "HiFi-GAN":
        # HIFIGAN_CONFIG = '../MG-Data/hifigan_ckpt/config.json'
        # HIFIGAN_CHECKPT = '../MG-Data/hifigan_ckpt/generator_universal.pth.tar'
        HIFIGAN_CONFIG_UN = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/config.json'
        HIFIGAN_CHECKPT_UN = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/g_02500000'
        HIFIGAN_CONFIG_EN = '../MG-Data/hifigan_ckpt/EN/config.json'
        HIFIGAN_CHECKPT_EN = '../MG-Data/hifigan_ckpt/EN/generator_LJSpeech.pth.tar'
        if speaker == "LJSpeech":
            with open(HIFIGAN_CONFIG_EN, "r") as f:
                config = json.load(f)
        elif speaker == "universal":
            with open(HIFIGAN_CONFIG_UN, "r") as f:
                config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(HIFIGAN_CHECKPT_EN)
        elif speaker == "universal":
            ckpt = torch.load(HIFIGAN_CHECKPT_UN)
        vocoder.load_state_dict(ckpt["generator"])
        _ = vocoder.cuda().eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "HiFi-GAN":
            wavs = vocoder.forward(mels)

        wavs = ( wavs.cpu().clamp(-1, 1).numpy() * preprocess_config["preprocessing"]["audio"]["max_wav_value"] ).astype(np.int16)
        # wavs = [wav for wav in wavs]

    return wavs
