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
from text import text_to_sequence, cmudict
import yaml
from model.gradtts import GradTTS
from utils.tools import to_device
import random
from tqdm import tqdm
from phonemizer.backend import EspeakBackend
backend = EspeakBackend(language='fr-fr')
from phonemizer.backend import EspeakMbrolaBackend
backend2 = EspeakMbrolaBackend(language='mb-fr2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HIFIGAN_CONFIG = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/config.json'
# HIFIGAN_CHECKPT = '../MG-Data/hifigan_ckpt/UNIVERSAL_V1/g_02500000'
# HIFIGAN_CONFIG = './hifigan/config.json'
# HIFIGAN_CHECKPT = './hifigan/generator_universal.pth.tar.zip'
# HIFIGAN_CONFIG = './LJ_FT_T2_V3/config.json'
# HIFIGAN_CHECKPT = './LJ_FT_T2_V3/generator_v3'
HIFIGAN_CONFIG = '../MG-Data/hifigan_ckpt/EN/config.json'
HIFIGAN_CHECKPT = '../MG-Data/hifigan_ckpt/EN/generator_LJSpeech.pth.tar'


# def intersperse(lst, item):
#     # Adds blank symbol
#     result = [item] * (len(lst) * 2 + 1)
#     result[1::2] = lst
#     return result


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
def preprocess_fr(text):
    # phones = []
    phonemized = backend.phonemize([text])
    phones = phonemized
    print(phones)
    return phones[0]


def synthesize_one_sample(args, result_path, text, spk, configs, i=0):
    preprocess_config, model_config, train_config = configs
    language = preprocess_config["preprocessing"]["text"]["language"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    add_blank = preprocess_config["preprocessing"]["g2p"]["add_blank"]
    dict_path = preprocess_config["preprocessing"]["g2p"]["dict_path"]
    if language == "ch":
        text = preprocess_mandarin(text)
        phones = np.array(
            text_to_sequence(language, True, text, cleaners))
    elif language == "en":
        dict = cmudict.CMUDict(dict_path)
        phones = np.array(
            text_to_sequence(language, False, text, cleaners, dict))
    else:
        # if language == "fr":
        text = preprocess_fr(text)
        phones = np.array(
            text_to_sequence(language, True, text, cleaners))

    if language == 'en':
        symbols_length = len(symbols_en)
    elif language == 'fr':
        symbols_length = len(symbols_fr)
    elif language == 'ch':
        symbols_length = len(symbols_ch)

    if add_blank:
        result = [symbols_length] * (len(phones) * 2 + 1)
        result[1::2] = phones
        phones = np.array([result])

    ids = raw_texts = [text[:100]]
    text_lens = np.array([phones.shape[-1]])
    spk = spk = np.array([random.randint(0,217)])
    speakers = spk

    t = dt.datetime.now()
    batchs = [(ids, raw_texts, speakers, phones, text_lens, max(text_lens))]
    batch = to_device(batchs[0], device)
    # (
    #     mu_x,
    #     logw,
    #     x_mask,
    #     y_mask,
    #     attn,
    #     mu_y,
    #     mels,
    #     encoder_outputs,
    #     decoder_outputs
    # ) 
    outputs = generator(*(batch[2:]), n_timesteps=args.timesteps, temperature=1, stoc=False, length_scale=1)
    t = (dt.datetime.now() - t).total_seconds()
    sample_rate = 22050
    # print(f'Grad-TTS RTF: {t * sample_rate / (outputs[8].shape[-1] * 256)}')
    audio = (vocoder.forward(outputs[8]).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

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
    parser.add_argument(
        '-dir',
        '--result_dir',
        type=str,
        required=False,
        default='0323-DDP',
        help='number of timesteps of reverse diffusion'
    )

    args = parser.parse_args()
    # with open('./syn_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

    preprocess_config_path = os.path.join("config", args.dataset, "preprocess.yaml")
    model_config_path = os.path.join("config", args.dataset, "model.yaml")
    train_config_path = os.path.join("config", args.dataset, "train.yaml")

    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    if preprocess_config["preprocessing"]["spk"]["n_spks"] > 1:
        assert args.speaker_id is not None
        spk = np.array([args.speaker_id])
    else:
        assert args.speaker_id is None
        spk = np.array([0])

    # Check source texts
    # if args.mode == "batch":
    #     assert args.source is not None and args.text is None
    # elif args.mode == "single":
    #     assert args.source is None and args.text is not None

    print('Initializing Grad-TTS...')

    generator = GradTTS(preprocess_config, model_config).to(device)
    # generator = get_model(args, configs, device, train=False).to(device)
    # generator = torch.nn.DataParallel(generator)
    checkpoint = torch.load(
            os.path.join(
                train_config["path"]["ckpt_path"], preprocess_config["dataset"], args.ckpt, f'{args.restore_epoch}.pt'
            ) , map_location=lambda loc, storage: loc
        )
    generator.load_state_dict(checkpoint)
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
    add_blank = preprocess_config["preprocessing"]["g2p"]["add_blank"]

    if language == "fr":
        symbols = symbols_fr
    elif language == "en":
        symbols = symbols_en
    elif language == "ch":
        symbols = symbols_ch

    result_path = os.path.join(train_config['path']['result_path'], args.dataset, args.result_dir)
    os.makedirs(result_path, exist_ok=True)
    # Preprocess texts
    if args.mode == "batch":
        with open(os.path.join('resources/filelists', args.source), 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        with torch.no_grad():
            for i, text in tqdm(enumerate(texts)):
                synthesize_one_sample(args, result_path, text, spk, configs, i)

    if args.mode == "single":
        with torch.no_grad():
            synthesize_one_sample(args, result_path, text, spk, configs)
