import sys
with open(sys.argv[1], "r") as f_src:
    src_sentences = f_src.readlines()

with open(sys.argv[2], "r") as f_tgt:
    tgt_sentences = f_tgt.readlines()

error_index = []

for i in range(len(src_sentences)):
    # print(sent.split(" "))
    src_sentences[i] = src_sentences[i].strip()
    tgt_sentences[i] = tgt_sentences[i].strip()
    # tgt_sentences[i] = tgt_sentences[i][:-1]
    if len(src_sentences[i].split(" ")) != len(tgt_sentences[i].split(" ")):
        print(src_sentences[i])
        print(len(src_sentences[i].split(" ")))
        print("---")
        print(tgt_sentences[i])
        print(len(tgt_sentences[i].split(" ")))
        error_index.append(i + 1)

with open("error_index.txt", "w") as f:
    for i in error_index:
        f.write(str(i) + "\n")

#sent = sent.replace('\u200b', '').strip()
