import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt


def synth_one_sample(idx, targets, predictions, vocoder, model_config, preprocess_config, path):

    basename = targets[0][idx]
    mel_len = targets[7][idx]
    mel_target = targets[6][idx,:,:mel_len].unsqueeze(0)# .detach()
    mel_prediction = predictions[8]# .detach()

    fig, data = plot_tensor(mel_target.squeeze().cpu())
    fig_prediction, data_prediction = plot_tensor(mel_prediction.squeeze().cpu())
    save_plot(mel_target.squeeze().cpu(), os.path.join(path, "{}_gt.png".format(basename[:-4])))
    save_plot(mel_prediction.squeeze().cpu(), os.path.join(path, "{}.png".format(basename[:-4])))

    if vocoder is not None:
        from utils.mymodel import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target,
            vocoder,
            model_config,
            preprocess_config,
        )
        wav_prediction = vocoder_infer(
            mel_prediction,
            vocoder,
            model_config,
            preprocess_config,
        )
    else:
        wav_reconstruction = wav_prediction = None

    return fig, fig_prediction, wav_reconstruction, wav_prediction, basename[:-4]


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        mel_len = predictions[6][i].size()[-1]
        mel_prediction = predictions[8][i, :mel_len].detach().transpose(0, 1)
        

        fig = plot_tensor(mel_prediction.squeeze().cpu())
        save_plot(mel_prediction.squeeze().cpu(), os.path.join(path, "{}.png".format(basename)))

    from utils.mymodel import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)

def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return fig, data

def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

