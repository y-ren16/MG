import torch
import torch.nn as nn
import os
import json
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from utils.tools import fix_len_compatibility, sequence_mask, generate_path
from model import monotonic_align
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradTTS(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(GradTTS, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config
        self.n_feats = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.n_spks = preprocess_config["preprocessing"]["spk"]["n_spks"]
        self.spk_emb_dim = preprocess_config["preprocessing"]["spk"]["spk_emb_dim"]

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                self.spk_emb_dim,
            )
        self.encoder = TextEncoder(preprocess_config, model_config)
        self.decoder = Diffusion(preprocess_config, model_config)

    def forward(
            self,
            speaker,
            texts,
            text_lens,
            max_text_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            out_size=None,
            n_timesteps=50,
            temperature=1.0,
            stoc=False,
            length_scale=1.0
        ):

        if self.n_spks > 1:
            speaker = self.speaker_emb(speaker)
        else:
            speaker = None

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(texts, text_lens, speaker)

        if mels is None:
            with torch.no_grad():
                w = torch.exp(logw) * x_mask
                w_ceil = torch.ceil(w) * length_scale
                y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
                y_max_length = int(y_lengths.max())
                y_max_length_ = fix_len_compatibility(y_max_length)

                # Using obtained durations `w` construct alignment map `attn`
                y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
                # torch.Size([16, 1, 119, 216])

                mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
                mu_y = mu_y.transpose(1, 2)
                encoder_outputs = mu_y[:, :, :y_max_length]

                decoder_outputs = self.decoder(y_mask, mu_y, temperature=temperature, n_timesteps=n_timesteps, stoc=stoc, spk=speaker)
                decoder_outputs = decoder_outputs[:, :, :y_max_length]
                return (
                    mu_x,
                    logw,
                    x_mask,
                    y_mask,
                    attn,
                    mu_y,
                    mels,
                    encoder_outputs,
                    decoder_outputs
                )
        else:
            y_lengths = mel_lens
            y_max_length = max_mel_len
            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), mels ** 2)
                # 每行一样
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), mels)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()
                attn_bp = attn

            # 截断最长172 随机偏移一定数值
            if not isinstance(out_size, type(None)):
                max_offset = (mel_lens - out_size).clamp(0)
                offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
                out_offset = torch.LongTensor([
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]).to(mel_lens)

                attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
                y_cut = torch.zeros(mels.shape[0], self.n_feats, out_size, dtype=mels.dtype, device=device)
                y_cut_lengths = []
                for i, (y_, out_offset_) in enumerate(zip(mels, out_offset)):
                    y_cut_length = out_size + (mel_lens[i] - out_size).clamp(None, 0)
                    y_cut_lengths.append(y_cut_length)
                    cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                    y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                    attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                y_cut_lengths = torch.LongTensor(y_cut_lengths)
                y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)

                attn = attn_cut
                mels = y_cut
                y_mask = y_cut_mask

            # Align encoded text and get mu_y
            # torch.Size([16, 216, 119])*torch.Size([16, 119, 80])=torch.Size([16, 216, 80])
            # attn 216对应那个119的emb
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
            mu_y = mu_y.transpose(1, 2)
            encoder_outputs = mu_y

            noise_estimation, z = self.decoder(y_mask, mu_y, mels=mels, n_timesteps=n_timesteps, stoc=stoc, spk=speaker)

            return (
                mu_x,
                logw,
                x_mask,
                y_mask,
                attn_bp,
                mu_y,
                mels,
                encoder_outputs,
                noise_estimation,
                z
            )

