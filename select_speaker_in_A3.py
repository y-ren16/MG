import os
speaker = ["SSB1781", "SSB1274", "SSB1585", "SSB1055", "SSB1020", "SSB0668", "SSB1625", "SSB0966", "SSB0760", "SSB1452"]
new_data = []
with open("../MG-Data/preprocessed_data/AISHELL3/train.txt", "r") as f:
    spk = []
    for line in f.readlines():
        data_col = line.strip("\n").split("|")
        spk = data_col[1]
        if spk in speaker:
            new_data.append(line)
    f.close()
with open("../MG-Data/preprocessed_data/AISHELL3/fake.txt", "w") as f:
    for line in new_data:
        f.write(line)
    f.close()