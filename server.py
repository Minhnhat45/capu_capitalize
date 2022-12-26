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
from flask import Flask, request, jsonify
import json
app = Flask(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = './model-bin/vlsp/'

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


def load_model(use_cuda):
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
    # use_cuda = True
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


@app.route('/capu', methods=['POST'])
def infer_sent():
    chunk_size = 15
    overlap_size = int(chunk_size / 2)
    full_sentence = ""
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

    sentence = request.json["input_string"]
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
        return json.dumps({"output_string": ' '.join(out_label_norm).strip()}, ensure_ascii=False)
        #print(' '.join(out_label_norm).strip())
    else:
        sub_list = []
        for sent in overlap:
            sub_list.append(sent.strip())
        merging_list = []
        for i, sentence in enumerate(sub_list):
            # print(sentence)
            # merging_list = []

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
        return json.dumps({"output_string": full_sentence}, ensure_ascii=False)


if __name__ == '__main__':
    #load model
    task_er, model_er, is_cuda = load_model(use_cuda=True)

    #infer one sentence overlap
    app.run(host='0.0.0.0', port=4445)
