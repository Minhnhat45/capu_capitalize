from sklearn.metrics import classification_report

import sys


def evaluate_result(file_truth, file_out):
    with open(file_truth, 'r', encoding='utf-8') as file_t:
        gr_truth = file_t.readlines()
    with open(file_out, 'r', encoding='utf-8') as file_o:
        hyp = file_o.readlines()
    list_truth = ""
    list_out = ""
    for i, (truth, out) in enumerate(zip(gr_truth, hyp)):
        assert len(truth.split()) == len(out.split()), "\n" + truth + str(len(truth.split())) + "\n" + out + str(len(out.split())) + "\n" + str(i)
        list_truth += truth + " "
        list_out += out + " "

    #classify_report = classification_report(list_truth.split(), list_out.split(), labels=['L,', 'L.', 'L?', 'T$', 'T,', 'T.', 'T?', 'U$', 'U,', 'U.', 'U?'])
    classify_report = classification_report(list_truth.split(), list_out.split(), labels=['L,', 'L.', 'L?', 'T$', 'T,', 'T.', 'U$', 'U,', 'U.'])
    print(classify_report)

evaluate_result(sys.argv[1], sys.argv[2])
