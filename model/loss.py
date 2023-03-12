import torch
import torch.nn as nn
from utils.tools import sequence_mask
import math
from model import monotonic_align
import random


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradLoss(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(GradLoss, self).__init__()
        self.n_feats = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.out_size = model_config["out_size"]

    def forward(self, inputs, predictions):
        (
            speaker,
            texts,
            text_lens,
            max_text_lens,
            mels,
            mel_lens,
            max_mel_len,
        ) = inputs[2:]
        (
            mu_x,
            logw,
            x_mask,
            y_mask,
            attn,
            mu_y,
            melsout,
            encoder_outputs,
            noise_estimation,
            z
        ) = predictions

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = torch.sum((logw - logw_) ** 2) / torch.sum(text_lens)

        diff_loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(y_mask) * self.n_feats)

        prior_loss = torch.sum(0.5 * ((melsout - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        total_loss = sum([dur_loss, prior_loss, diff_loss])
        # total_loss = (
        #         dur_loss + prior_loss + diff_loss
        # )

        return total_loss, dur_loss, prior_loss, diff_loss






