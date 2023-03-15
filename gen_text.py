import os
import random
texts = []
i = 0
with open('./resources/filelists/test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if len(line) > 50 and i < 1000:
            texts.append(line.replace(" ", ""))
            i += 1
        elif len(line) > 20 and i < 1500 and i >= 1000:
            texts.append(line.replace(" ", ""))
            i += 1
    random.shuffle(texts)

with open('./resources/filelists/synthetic.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        f.write(text)
