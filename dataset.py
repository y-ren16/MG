import os
import numpy as np
import torch
from torch.utils.data import Dataset
from text import text_to_sequence, cmudict
from text.symbols import symbols_fr, symbols_en, symbols_ch
import torchaudio
from hifigan import mel_spectrogram
from utils.tools import pad_1D, pad_2D
import json


class TextMelDataset(Dataset):
    def __init__(
            self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]

        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.wav_path = preprocess_config["path"]["corpus_path"]

        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.language = preprocess_config["preprocessing"]["text"]["language"]

        self.batch_size = train_config["optimizer"]["batch_size"]

        self.add_blank = preprocess_config["preprocessing"]["g2p"]["add_blank"]
        self.dict_path = preprocess_config["preprocessing"]["g2p"]["dict_path"]

        self.sample_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

        self.n_fft = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.win_length = preprocess_config["preprocessing"]["stft"]["win_length"]

        self.n_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.f_min = preprocess_config["preprocessing"]["mel"]["mel_fmin"]
        self.f_max = preprocess_config["preprocessing"]["mel"]["mel_fmax"]

        self.n_spks = preprocess_config["preprocessing"]["spk"]["n_spks"]
        if self.n_spks > 1:
            with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
                self.speaker_map = json.load(f)

        self.name_col = preprocess_config["preprocessing"]["dataform"]["name_col"]
        self.speaker_col = preprocess_config["preprocessing"]["dataform"]["speaker_col"]
        self.phone_col = preprocess_config["preprocessing"]["dataform"]["phone_col"]
        self.raw_text_col = preprocess_config["preprocessing"]["dataform"]["raw_text_col"]

        if self.language == 'en':
            self.dict = cmudict.CMUDict(self.dict_path)

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.basename)

    def process_meta(self, filename):
        with open(
                os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                data_col = line.strip("\n").split("|")
                name.append(data_col[self.name_col])
                if self.speaker_col < 100:
                    speaker.append(data_col[self.speaker_col])
                else:
                    speaker.append(self.dataset_name)
                if self.phone_col < 100:
                    text.append(data_col[self.phone_col])
                raw_text.append(data_col[self.raw_text_col])
            return name, speaker, text, raw_text

    def __getitem__(self, index):
        basename = self.basename[index]
        if basename[-4:] != '.wav':
            basename += '.wav'

        if self.n_spks > 1:
            speaker = self.speaker[index]
            speaker_id = self.speaker_map[speaker]
            speaker = int(speaker_id)
        else:
            speaker = 0

        raw_text = self.raw_text[index]

        if self.language == 'en':
            symbols_length = len(symbols_en)
        elif self.language == 'en':
            symbols_length = len(symbols_fr)
        elif self.language == 'ch':
            symbols_length = len(symbols_ch)


        if self.phone_col < 100:
            phone = np.array(text_to_sequence(self.language, True, self.text[index], self.cleaners))
        else:
            phone = np.array(text_to_sequence(self.language, False, self.raw_text[index], self.cleaners, self.dict))

        if self.add_blank:
            result = [symbols_length] * (len(phone) * 2 + 1)
            result[1::2] = phone
            phone = result
            # add a blank token, whose id number is len(symbols)

        phone = torch.LongTensor(phone)
        audio, sr = torchaudio.load(
            os.path.join(self.wav_path, basename)
        )
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()


        sample = {
            "id": basename,
            "speaker": speaker,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel.T,
        }

        return sample

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        speakers = np.array(speakers)

        def fix_len_compatibility(length, num_downsamplings_in_unet=2):
            while True:
                if length % (2 ** num_downsamplings_in_unet) == 0:
                    return length
                length += 1

        max_mel_len = max(mel_lens)
        max_mel_len = fix_len_compatibility(max_mel_len)

        texts = pad_1D(texts)
        mels = pad_2D(mels, max_mel_len)

        # texts = torch.LongTensor(texts)
        # mels = torch.Tensor(mels)
        # speakers = torch.LongTensor(speakers)
        mels1 = np.transpose(mels, axes=(0, 2, 1))

        max_text_lens = max(text_lens)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max_text_lens,
            mels1,
            mel_lens,
            max_mel_len,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    preprocess_config = yaml.load(
        open("./config/BC2023/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/BC2023/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = TextMelDataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = TextMelDataset(
        "valid.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
