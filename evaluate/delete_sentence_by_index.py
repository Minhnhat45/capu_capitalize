from sys import argv

with open(argv[1], mode="r") as input_file:
    texts = input_file.readlines()

with open(argv[2], mode="r") as index_file:
    indices = [int(i) for i in index_file.readlines()]
    indices.sort()

print(indices)
count = 0
for nu in indices:
    del texts[nu - count - 1]
    count += 1

with open(argv[3], mode="w") as output_file:
    for text in texts:
        output_file.writelines(text)
