#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from fairseq import checkpoint_utils, data, options, tasks
import utils as tagging_utils
from fairseq import utils
import torch
import functools
import os
from plugin.data import normalize_dataset
from plugin.tasks import normalize
import time
from seqeval.metrics import f1_score, classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = 'model-bin/vlsp/'

def infer(inputs, task, model, use_cuda, batch_size=1):
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:1 + n]

    results = [[]]
    results_plain = []
    words = "".join(inputs).split()

    words_length = [1]
    words_unsub_length = [1]

    for w in words:
        w_pieces = task.trained_lm.bpe.sp.EncodeAsPieces(w)
        words_length.extend([len(w_pieces)] + [0] * (len(w_pieces) - 1))
        words_unsub_length.extend([len(w_pieces)])
    words_length.append(1)
    source_word_size = torch.tensor(words_length)
    words_unsub_length.append(1)
    word_encode = torch.tensor(words_unsub_length)

    for sentences in chunks(inputs, batch_size):
        for idx, sen in enumerate(sentences):
            samples_list = [
                {
                    'id': idx,
                    'source': task.trained_lm.encode(sen.strip()),
                    # 'source_plain': task.trained_lm.bpe.sp.EncodeAsPieces(sen),
                    'source_plain': sen.split(" "),
                    'sen': sen,
                    'source_word_size': source_word_size,
                    'word_encode': word_encode
                }
            ]
        batch = normalize_dataset.collate(
            samples_list, pad_idx=0, eos_idx=task.source_dictionary.eos(),
            left_pad_source=False, left_pad_target=False,
            input_feeding=True
        )
        batch = utils.move_to_cuda(batch) if use_cuda else batch
        pair_preds = model.encode(**batch['net_input'])

        for i, (preds, sample) in enumerate(zip(pair_preds, samples_list)):
            result_encoded = task.target_dictionary.string(preds).split()
            output_plain = []
            for sub_word, label in zip(sample['source_plain'], result_encoded):
                # word_nor = sub_word
                def normalize_tag(sub_word_nor, label_nor):
                    def check_punctuation(sub_word_punc, label_punc):
                        if label_punc[1] == "," or label_punc[1] == "." or label_punc[1] == "?":
                            sub_word_punc += label_punc[1]
                            return sub_word_punc
                        else:
                            return sub_word_punc

                    if label_nor[0] == "T":
                        sub_word_nor = sub_word_nor.capitalize()
                        sub_word_nor = check_punctuation(sub_word_nor, label_nor)
                    elif label_nor[0] == "U":
                        sub_word_nor = sub_word_nor.upper()
                        sub_word_nor = check_punctuation(sub_word_nor, label_nor)
                    else:
                        sub_word_nor = sub_word_nor
                        sub_word_nor = check_punctuation(sub_word_nor, label_nor)
                    return sub_word_nor

                output_plain.append('{} {}'.format(sub_word, normalize_tag(sub_word, label)))
                results[i].append(normalize_tag(sub_word, label))
            results_plain.append(' '.join(output_plain))


    return results, results_plain

def infer_to_tag(inputs, task, model, use_cuda, batch_size=1):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    results = [[]]
    results_plain = []
    words = "".join(inputs).split()
    #print(words)

    words_length = [1]
    words_unsub_length = [1]

    for w in words:
        w_pieces = task.trained_lm.bpe.sp.EncodeAsPieces(w)
        words_length.extend([len(w_pieces)] + [0] * (len(w_pieces) - 1))
        words_unsub_length.extend([len(w_pieces)])
    words_length.append(1)
    source_word_size = torch.tensor(words_length)
    words_unsub_length.append(1)
    word_encode = torch.tensor(words_unsub_length)

    for sentences in chunks(inputs, batch_size):
        for idx, sen in enumerate(sentences):
            samples_list = [
                {
                    'id': idx,
                    'source': task.trained_lm.encode(sen.strip()),
                    #'source_plain': task.trained_lm.bpe.sp.EncodeAsPieces(sen),
                    'source_plain': sen.split(" "),
                    'sen': sen,
                    'source_word_size': source_word_size,
                    'word_encode': word_encode
                }
            ]
        #print(samples_list)
        # Build mini-batch to feed to the model
        # pad_idx=task.source_dictionary.pad()
        batch = normalize_dataset.collate(
            samples_list, pad_idx=0, eos_idx=task.source_dictionary.eos(),
            left_pad_source=False, left_pad_target=False,
            input_feeding=True
        )

        # Feed batch to the model and get predictions
        batch = utils.move_to_cuda(batch) if use_cuda else batch
        pair_preds = model.encode(**batch['net_input'])
        for i, (preds, sample) in enumerate(zip(pair_preds, samples_list)):
            result_encoded = task.target_dictionary.string(preds).split()
            output_plain = []
            for sub_word, label in zip(sample['source_plain'], result_encoded):
                output_plain.append('{} {}'.format(sub_word, label))
                results[i].append(label)
            results_plain.append(' '.join(output_plain).replace('▁', ''))
            assert len(results[i]) == len(sample['sen'].split()), "{}\n{}\n{}\n{}".format(sample['sen'], results[i],
                                                                                          len(results[i]),
                                                                                          len(sample['sen'].split()))
    return results, results_plain



def load_model():
    tagging_utils.import_user_module('./plugin')
    sys.argv += [
        os.path.join(model_path, 'dict'),
        '--user-dir', './plugin',
        '--task', 'normalize',
        '-s', 'src', '-t', 'tgt',
        '--max-tokens', '4000',
        '--trained-lm', model_path,
        '--rnn-type', 'GRU',
        '--rnn-layers', '4',
    ]

    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    utils.import_user_module(args)
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    #use_cuda = torch.cuda.is_available() and not args.cpu
    use_cuda = True
    #print(args)

    task_normalize = tasks.setup_task(args)

    print('| loading model from {}'.format(os.path.join(model_path, 'checkpoint_best.pt')))
    models, _model_args = checkpoint_utils.load_model_ensemble([os.path.join(model_path, 'checkpoint_best.pt')],
                                                               task=task_normalize)
    model_normalize = models[0]
    model_normalize.eval()
    if use_cuda:
        model_normalize.cuda()

    #print(model_normalize)
    return task_normalize, model_normalize, use_cuda

def pretty_format(raw_input):
    norm_input = raw_input[0].split()
    in_w, out_label_norm = [], []
    for i, item_norm in enumerate(zip(norm_input)):
        if i % 2 == 0:
            in_w.append("%13s" % item_norm)
        else:
            out_label_norm.append("%13s" % item_norm)
    return "{}\n{}".format(''.join(in_w), ''.join(out_label_norm))
def evaluate_result(file_truth, file_out):
    from sklearn.metrics import classification_report
    with open(file_truth, 'r', encoding='utf-8') as file_t:
        with open(file_out, 'r', encoding='utf-8') as file_o:
            list_truth = []
            list_output = []
            for truth, out in zip(file_t, file_o):
                assert len(truth.split()) == len(out.split()), truth + str(len(truth.split())) + "\n" + out + str(
                    len(out.split())) + "\n\n"
                list_truth.append(truth.split())
                list_output.append(out.split())

            print(list_truth)
            print(list_output)

            classify_report = classification_report(list_truth, list_output)
            print(classify_report)

if __name__ == '__main__':
    #load model
    task_er, model_er, is_cuda = load_model()

    #infer one sentence overlap
    chunk_size = 15
    overlap_size = int(chunk_size / 2)
    def overlap_cut(sentence):
        list_word = sentence.split(" ")
        start_index = 0
        sub_sentences = []
        overlap_time = 0
        if len(list_word[start_index:start_index + chunk_size]) >= chunk_size:
            while len(list_word[start_index:start_index + chunk_size]) >= chunk_size:
                overlap_time += 1
                sub_sentences.append(" ".join(list_word[start_index: start_index + chunk_size]))
                start_index += overlap_size
            all_len = len(" ".join(sub_sentences).split())
            list_word_used = all_len - (overlap_time - 1) * (chunk_size - overlap_size)
            words_left_over = len(list_word) - list_word_used
            # print(words_left_over)
            if words_left_over != 0:
                sub_sentences[-1] += " " + " ".join(list_word[list_word_used:len(list_word)])
            return sub_sentences
        else:
            return sentence

    def merging(head_sentence, sub_sentence):
        min_words_cut = int((chunk_size - overlap_size) / 2)
        sentence = head_sentence.strip().split()[:-min_words_cut] + sub_sentence.strip().split()[min_words_cut:]
        return " ".join(sentence)

    def remove_punc_selection(sentence, out_sentence):
        def remove_punc(word):
            import string
            word = word.translate(str.maketrans('', '', string.punctuation))
            return word

        list_out = []
        for i in range(0, len(sentence.split())):
            if len(sentence.split()[i]) < len(out_sentence.split()[i]):
                print(sentence.split()[i])
                print(out_sentence.split()[i])
                if len(out_sentence.split()[i]) - len(remove_punc(out_sentence.split()[i])) <= 1:
                    list_out.append(remove_punc(out_sentence.split()[i]))
                else:
                    out_word = out_sentence.split()[i][:-1]
                    list_out.append(out_word)
            else:
                list_out.append(out_sentence.split()[i])


        # sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        print(list_out)
        return " ".join(list_out)

    while True:
        sentence = input('\nInput: ')
        # dict = {dot = [], comma = [], question = []}
        start_time = time.time()
        if len(sentence.split()) > 1.5 * chunk_size:
            overlap = overlap_cut(sentence.strip())
        else:
            overlap = sentence.strip()
        if isinstance(overlap, str):
            _, model_output = infer([overlap], task_er, model_er, is_cuda)
            norm_input = model_output[0].split()
            in_w, out_label_norm = [], []
            for j, item_norm in enumerate(norm_input):
                if j % 2 == 0:
                    in_w.append("%13s" % item_norm)
                else:
                    out_label_norm.append("%s" % item_norm)
            output_sent = ' '.join(out_label_norm).strip()
            print(remove_punc_selection(sentence, output_sent))
        else:
            sub_list = []
            for sent in overlap:
                sub_list.append(sent.strip())
            merging_list = []
            for i, sentence in enumerate(sub_list):
                # print(sentence)
                # merging_list = []
                full_sentence = ""
                if len(sentence) > 0:
                    _, model_output = infer([sentence], task_er, model_er, is_cuda)
                    norm_input = model_output[0].split()
                    in_w, out_label_norm = [], []
                    for j, item_norm in enumerate(norm_input):
                        if j % 2 == 0:
                            in_w.append("%13s" % item_norm)
                        else:
                            out_label_norm.append("%s" % item_norm)
                    merging_list.append(' '.join(out_label_norm).strip())

            full_sentence = merging_list[0]
            # print(len(merging_list))
            for i in range(1, len(merging_list)):
                full_sentence = merging(full_sentence, merging_list[i])

            print(remove_punc_selection(sentence, full_sentence))
        print("\nTime infer: {}s".format(time.time() - start_time))

    #infer by file
    # with open("evaluate/eval_indomain_full.src", "r") as f:
    #     sentences = f.readlines()
    # start_time = time.time()
    # with open("evaluate/eval_indomain_full_1.hyp", "w") as f:
    #     for i, sentence in enumerate(sentences):
    #         print(i)
    #         if len(sentence) > 0:
    #             _, model_output = infer_to_tag([sentence], task_er, model_er, is_cuda)
    #             norm_input = model_output[0].split()
    #             in_w, out_label_norm = [], []
    #             for i, item_norm in enumerate(zip(norm_input)):
    #                 if i % 2 == 0:
    #                     in_w.append("%13s" % item_norm)
    #                 else:
    #                     out_label_norm.append("%s" % item_norm)
    #             f.write(' '.join(out_label_norm).strip() + "\n")
    # print("\nTime infer: {}s".format(time.time() - start_time))

    #infer overlap
    # chunk_size = 15
    # overlap_size = int(chunk_size / 2)
    # def overlap_cut(sentence):
    #     list_word = sentence.split(" ")
    #     start_index = 0
    #     sub_sentences = []
    #     overlap_time = 0
    #     if len(list_word[start_index:start_index + chunk_size]) >= chunk_size:
    #         while len(list_word[start_index:start_index + chunk_size]) >= chunk_size:
    #             overlap_time += 1
    #             sub_sentences.append(" ".join(list_word[start_index: start_index + chunk_size]))
    #             start_index += overlap_size
    #         all_len = len(" ".join(sub_sentences).split())
    #         list_word_used = all_len - (overlap_time - 1) * (chunk_size - overlap_size)
    #         words_left_over = len(list_word) - list_word_used
    #         # print(words_left_over)
    #         if words_left_over != 0:
    #             sub_sentences[-1] += " " + " ".join(list_word[list_word_used:len(list_word)])
    #         return sub_sentences
    #     else:
    #         return sentence


    # def merging(head_sentence, sub_sentence):
    #     min_words_cut = int((chunk_size - overlap_size) / 2)
    #     sentence = head_sentence.strip().split()[:-min_words_cut] + sub_sentence.strip().split()[min_words_cut:]
    #     return " ".join(sentence)
    # with open("evaluate/eval_indomain_full.src", "r") as f:
    #     sentences = f.readlines()
    # overlap_list = []
    # for i, sentence in enumerate(sentences):
    #     print(i)
    #     if len(sentence.split()) > 1.5 * chunk_size:
    #         overlap = overlap_cut(sentence.strip())
    #     else:
    #         overlap = sentence.strip()
    #     if isinstance(overlap, str):
    #         _, model_output = infer_to_tag([overlap], task_er, model_er, is_cuda)
    #         norm_input = model_output[0].split()
    #         in_w, out_label_norm = [], []
    #         for j, item_norm in enumerate(norm_input):
    #             if j % 2 == 0:
    #                 in_w.append("%13s" % item_norm)
    #             else:
    #                 out_label_norm.append("%s" % item_norm)
    #         overlap_list.append(' '.join(out_label_norm).strip())
    #     else:
    #         sub_list = []
    #         for sent in overlap:
    #             sub_list.append(sent.strip())
    #         merging_list = []
    #         for i, sentence in enumerate(sub_list):
    #             # print(sentence)
    #             # merging_list = []
    #             full_sentence = ""
    #             if len(sentence) > 0:
    #                 _, model_output = infer_to_tag([sentence], task_er, model_er, is_cuda)
    #                 norm_input = model_output[0].split()
    #                 in_w, out_label_norm = [], []
    #                 for j, item_norm in enumerate(norm_input):
    #                     if j % 2 == 0:
    #                         in_w.append("%13s" % item_norm)
    #                     else:
    #                         out_label_norm.append("%s" % item_norm)
    #                 merging_list.append(' '.join(out_label_norm).strip())

    #         full_sentence = merging_list[0]
    #         # print(len(merging_list))
    #         for i in range(1, len(merging_list)):
    #             full_sentence = merging(full_sentence, merging_list[i])
    #         overlap_list.append(full_sentence)
    # # print(len(overlap_list))
    # with open("evaluate/eval_indomain_full_1_merging.hyp", "w") as f:
    #     for sent in overlap_list:
    #         f.write(sent.strip() + "\n")
    # print(overlap_list)
    # for sent in overlap_list:
    #     print(len(sent.split()))

    #evaluate model
    #evaluate_result("evaluate/eval_outdomain.tgt", "evaluate/eval_outdomain.hyp")
