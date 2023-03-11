import re
import argparse
# from utils.mymodel import get_model
from text import symbols_fr, symbols_en, symbols_ch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import hifigan
import json
from scipy.io.wavfile import write
import datetime as dt
import numpy as np
from pypinyin import pinyin, Style
from text import text_to_sequence
import yaml
from model.gradtts import GradTTS
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIFIGAN_CONFIG = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/config.json'
HIFIGAN_CHECKPT = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/g_02500000'
# HIFIGAN_CONFIG = './hifigan/config.json'
# HIFIGAN_CHECKPT = './hifigan/generator_universal.pth.tar.zip'
# HIFIGAN_CONFIG = './LJ_FT_T2_V3/config.json'
# HIFIGAN_CHECKPT = './LJ_FT_T2_V3/generator_v3'


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_mandarin(text):
    lexicon = read_lexicon("./resources/pinyin-lexicon-r.txt")

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    # print(f"Raw Text Sequence: {text}")
    # print(f"Phoneme Sequence: {phones}")
    return phones


def synthesize_one_sample(args, result_path, text, spk, language, cleaners, i=0):
    if language == "ch":
        text = preprocess_mandarin(text)
        x = torch.LongTensor([
            intersperse(text_to_sequence(language, True, text, cleaners), len(symbols_ch))])
    elif language == "en":
        x = torch.LongTensor([
            intersperse(text_to_sequence(language, False, text, cleaners), len(symbols_en))])
    else:
        # if language == "fr":
        x = torch.LongTensor([
            intersperse(text_to_sequence(language, True, text, cleaners), len(symbols_fr))])

    ids = raw_texts = [text[:100]]
    text_lens = torch.LongTensor([x.shape[-1]])
    speakers = spk
    phones = x

    t = dt.datetime.now()
    batchs = [(ids, raw_texts, speakers, phones, text_lens, text_lens)]
    batch = to_device(batchs[0], device)
    (
        mu_x,
        logw,
        x_mask,
        y_mask,
        attn,
        mu_y,
        mels,
        encoder_outputs,
        decoder_outputs
    ) = generator(*(batch[2:]), n_timesteps=args.timesteps, temperature=1.5, stoc=False, length_scale=0.91)
    t = (dt.datetime.now() - t).total_seconds()
    sample_rate = 22050
    print(f'Grad-TTS RTF: {t * sample_rate / (decoder_outputs.shape[-1] * 256)}')
    audio = (vocoder.forward(decoder_outputs).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

    if args.speaker_id:
        write(
            f'{result_path}/sample_pt{str(args.restore_epoch)}_sp{args.speaker_id}_t{args.timesteps}_line{i}.wav',
            sample_rate, audio
        )
    else:
        write(
            f'{result_path}/sample_pt{str(args.restore_epoch)}_t{args.timesteps}_line{i}.wav',
            sample_rate, audio
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r',
        '--restore_epoch',
        type=int,
        # required=True,
        default=100,
        help='restore_epoch of Grad-TTS'
    )
    parser.add_argument(
        '-c',
        '--ckpt',
        type=str,
        # required=True,
        default="2023-03-11-02_53",
        help='path to a checkpoint of Grad-TTS'
    )
    parser.add_argument(
        '-d',
        "--dataset",
        type=str,
        # required=True,
        default="LJSpeech",
        help="dataset yaml",
    )
    parser.add_argument(
        '-m',
        "--mode",
        type=str,
        choices=["batch", "single"],
        # required=True,
        default="batch",
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        '-s',
        "--source",
        type=str,
        default="syn_en.txt",
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        '-t',
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        '-id',
        "--speaker_id",
        type=int,
        default=None,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        '-time',
        '--timesteps',
        type=int,
        required=False,
        default=50,
        help='number of timesteps of reverse diffusion'
    )

    args = parser.parse_args()

    preprocess_config_path = os.path.join("config", args.dataset, "preprocess.yaml")
    model_config_path = os.path.join("config", args.dataset, "model.yaml")
    train_config_path = os.path.join("config", args.dataset, "train.yaml")

    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    if not isinstance(args.speaker_id, type(None)):
        assert preprocess_config["spk"]["n_spks"] > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id])
    else:
        spk = torch.LongTensor([0])

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    elif args.mode == "single":
        assert args.source is None and args.text is not None

    print('Initializing Grad-TTS...')

    generator = GradTTS(preprocess_config, model_config).to(device)
    # generator = get_model(args, configs, device, train=False).to(device)
    generator.load_state_dict(
        torch.load(
            os.path.join(
                train_config["path"]["ckpt_path"], preprocess_config["dataset"], args.ckpt, f'{args.restore_epoch}.pt'
            ), map_location=lambda loc, storage: loc
        )
    )
    _ = generator.eval()

    print('Initializing HiFi-GAN...')

    with open(HIFIGAN_CONFIG, "r") as f:
        h = hifigan.AttrDict(json.load(f))

    vocoder = hifigan.Generator(h)
    device = torch.device('cuda')
    ckpt = torch.load(HIFIGAN_CHECKPT, map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    language = preprocess_config["preprocessing"]["text"]["language"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    if language == "fr":
        symbols = symbols_fr
    elif language == "en":
        symbols = symbols_en
    elif language == "ch":
        symbols = symbols_ch

    result_path = os.path.join(train_config['path']['result_path'], args.dataset)
    os.makedirs(result_path, exist_ok=True)
    # Preprocess texts
    if args.mode == "batch":
        with open(os.path.join('resources/filelists', args.source), 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        with torch.no_grad():
            for i, text in enumerate(texts):
                synthesize_one_sample(args, result_path, text, spk, language, cleaners, i)

    if args.mode == "single":
        with torch.no_grad():
            synthesize_one_sample(args, result_path, text, spk, language, cleaners)
