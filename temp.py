import os

file_name = ["train.txt", "test.txt", "valid.txt"]
for name in file_name:
    new = []
    with open(
            os.path.join("../Speech-Backbones-main/Grad-TTS/resources/filelists/ljspeech", name), "r", encoding="utf-8"
    ) as f:
        for line in f.readlines():
            new_line = line.replace("DUMMY/", "")
            new.append(new_line)
    with open(os.path.join("../MG-Data/preprocessed_data/LJSpeech", name), "w"
              ) as f1:
        for line in new:
            f1.write(line)
