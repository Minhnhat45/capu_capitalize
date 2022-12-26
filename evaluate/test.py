import sys
with open(sys.argv[1], "r") as f:
    target = f.readlines()

#with open("crawl_baochi_client.hyp", "r") as f:
#    client = f.readlines()


target_crop = []
#client_crop = []
for i, sent in enumerate(target):
    target_crop.append(" ".join(target[i].strip().split()[:-1]))
#    client_crop.append(" ".join(client[i].strip().split()[:-1]))
#print(" ".join(target[1].strip().split()[:-1]))

with open(sys.argv[2], "w") as f:
    for sent in target_crop:
        f.write(sent.strip() + "\n")

#with open("crawl_baochi_client_crop.hyp", "w") as f:
#    for sent in client_crop:
#        f.write(sent.strip() + "\n")
        
