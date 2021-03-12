import os
from os import listdir
from os.path import isfile, join

DIR = "."

# take all files

onlyfiles = [f for f in listdir(DIR) if isfile(join(DIR, f))]
os.system("mkdir temp")

for i, text_file in enumerate(onlyfiles):
    f = open(text_file, 'r')
    all_text = ""
    for line in f:
        line = line.strip()
        if len(line) > 0:
            all_text += line
    f.close()
    all_text = all_text.split(".")
    all_text = [sentence + "\n" for sentence in all_text]
    final = "".join(all_text)
    temp_output = open("temp/temp" + i, 'w')
    for sentence in all_text:
        temp_output.write(final)
    temp_output.close()

# combine all temp* files
temp_files = [f for f in listdir(DIR + "/temp") if isfile(join(DIR + "/temp", f))]
corpus = open("corpus.txt", 'w')
for temp in temp_files:
    corpus.write(temp.read() + "\n")
corpus.close()

