with open("eval_outdomain.src", "r") as f:
    eval_src = f.readlines()
with open("eval_outdomain.tgt", "r") as f:
    eval_tgt = f.readlines()


eval_nor_src = []
eval_nor_tgt = []

for i, sent in enumerate(eval_src):
    if len(sent) < 500:
        eval_nor_src.append(eval_src[i])
        eval_nor_tgt.append(eval_tgt[i])


with open("eval_outdomain_nor.src", "w") as f:
    for sent in eval_nor_src:
        f.write(sent.strip() + "\n")
with open("eval_outdomain_nor.tgt", "w") as f:
    for sent in eval_nor_tgt:
        f.write(sent.strip() + "\n")
