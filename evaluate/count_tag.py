import sys

with open(sys.argv[1], "r") as f:
    all_data = f.readlines()

tag_count = {'L$': 0,
             'L,': 0,
             'L.': 0,
             'L?': 0,
             'T$': 0,
             'T,': 0,
             'T.': 0,
             'T?': 0,
             'U$': 0,
             'U,': 0,
             'U.': 0,
             'U?': 0}

# print(tag_count['L$'])
# if 'L$' in tag_count.keys():
#     tag_count['L$'] += 10
#
# print(tag_count)
for sent in all_data:
    for tag in sent.split():
        if tag in tag_count.keys():
            tag_count[tag] += 1

print(tag_count)