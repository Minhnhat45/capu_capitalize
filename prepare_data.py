"""
Data pre-processing: build vocabularies and binarize training data.
"""
import utils as m_utils
from fairseq import options, tasks, utils
import os
from collections import Counter
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer, safe_readline
from multiprocessing import Pool
from utils import tokenize_line_char, tokenize_line_word
import torch
import glob
from tqdm import tqdm
import random
from fairseq.models.roberta import XLMRModel, RobertaHubInterface
import re
import numpy as np

DEV_RATIO = 0.01
TEST_RATIO = 0.01


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def str_to_bin(filename, dict, consumer, append_eos=True, reverse_order=False,
               offset=0, end=-1):
    nseq, ntok = 0, 0
    replaced = Counter()

    def replaced_consumer(word, idx):
        if idx == dict.unk_index and word != dict.unk_word:
            replaced.update([word])

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            if isinstance(dict, RobertaHubInterface):
                words = line.strip().split()

                words_length = [1]
                words_unsub_length = [1]
                # subwords_sent = []
                for w in words:
                    w_pieces = dict.bpe.sp.EncodeAsPieces(w)
                #     if len(w_pieces) > 1:
                #         for subword in w_pieces:
                #             subwords_sent.append(subword.replace("▁", ""))
                #     else:
                #         subwords_sent.append(str(w_pieces).replace("[\'", "").replace("▁", "").replace("\']", ""))

                    words_length.extend([len(w_pieces)] + [0] * (len(w_pieces) - 1))
                    words_unsub_length.extend([len(w_pieces)])
                words_length.append(1)
                words_unsub_length.append(1)
                # words_unsub_length = np.ones((len(words) + 2,), dtype=int).tolist()
                # create align matrix
                # align_matrix = np.zeros((len(words), len(subwords_sent)))
                # for i in range(len(words)):
                #     for j in range(len(subwords_sent)):
                #         if subwords_sent[j] in words[i]:
                #             align_matrix[i][j] = 1
                #             for k, sub in enumerate(subwords_sent[j + 1:]):
                #                 if sub in words[i]:
                #                     align_matrix[i][k + j + 1] = 1

                ids = dict.encode(line.strip())
                assert len(ids) == len(words_length), line
                ids = torch.cat([ids, torch.tensor(words_length)])
                ids_array = ids.numpy()
                # reshape_matrix = np.reshape(align_matrix, (1, -1))
                # ids = np.append(ids_array, reshape_matrix)
                ids = np.append(ids, words_unsub_length)
                ids = np.append(ids, len(words) + 2)
                ids = np.append(ids, len(words_length))
                # ids = np.append(ids_array, len(words))
                # ids = np.append(ids, len(words_length) - 2)
                ids = torch.tensor(ids)
            else:
                ids = dict.encode_line(
                    line=line,
                    line_tokenizer=tokenize_line_word,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
            nseq += 1
            ntok += len(ids)
            consumer(ids)
            line = f.readline()
    return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=False):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl,
                                      vocab_size=len(vocab) if not isinstance(vocab, RobertaHubInterface) else len(
                                          vocab.task.source_dictionary))

    def consumer(tensor):
        ds.add_item(tensor)

    res = str_to_bin(filename, vocab, consumer, append_eos=append_eos,
                     offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def prepare_raw_data(args, word_src_dict, word_tgt_dict):
    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, char_vocab=None):
        if isinstance(vocab, RobertaHubInterface):
            print("| [{}] Dictionary: {} types".format(lang, len(vocab.task.source_dictionary) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl,
                                          vocab_size=len(vocab) if not isinstance(vocab, RobertaHubInterface) else len(
                                              vocab.task.source_dictionary))
        merge_result(
            str_to_bin(
                input_file, vocab, lambda t: ds.add_item(t), char_vocab,
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by unk".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1]
            )
        )

    if args.trainpref:
        make_binary_dataset(word_src_dict, args.trainpref, "train", args.source_lang, num_workers=args.workers)
        make_binary_dataset(word_tgt_dict, args.trainpref, "train", args.target_lang, num_workers=args.workers)
    if args.validpref:
        make_binary_dataset(word_src_dict, args.validpref, "valid", args.source_lang, num_workers=args.workers)
        make_binary_dataset(word_tgt_dict, args.validpref, "valid", args.target_lang, num_workers=args.workers)


def prepare_dict(args, pre_trained_lm):
    utils.import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False, word_level=True):
        assert src ^ tgt
        return task.build_dict(
            filenames,
            word_level=word_level,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    if (os.path.exists(dict_path(args.source_lang)) or pre_trained_lm) and \
            os.path.exists(dict_path(args.target_lang)):
        if pre_trained_lm:
            input_dict = pre_trained_lm.task.source_dictionary
        else:
            input_dict = task.load_dictionary(dict_path(args.source_lang))
        label_dict = task.load_dictionary(dict_path(args.target_lang))
        return input_dict, label_dict

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    # char_dict = build_dictionary(
    #     {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True, word_level=False
    # )
    #
    # # print(src_dict)
    # char_dict.save(os.path.join(args.destdir, 'dict_char.txt'))
    return src_dict, tgt_dict


def cli_main(pre_trained_lm=None):
    parser = options.get_preprocessing_parser(default_task='normalize')
    args = parser.parse_args()
    src_dict, tgt_dict = prepare_dict(args, pre_trained_lm)
    if not pre_trained_lm:
        prepare_raw_data(args, src_dict, tgt_dict)
    else:
        prepare_raw_data(args, pre_trained_lm, tgt_dict)


def parse_raw_vlsp2016(file_path):
    sample_list = []
    sample = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == '<s>':
                sample = []
            elif line.strip() == '</s>' and sample:
                input_sent = [item[0] for item in sample]
                output_sent = [item[1] for item in sample]
                sample_list.append((input_sent, output_sent))
            else:
                line = line.strip().split('\t')
                if len(line) == 5:
                    word = re.sub(r'\s', '', line[0].strip()).split('_')
                    word = [item.strip() for item in word if item.strip()]
                    if word:
                        label = line[-2]
                        if not label.startswith('B'):
                            label = [label] * len(word)
                        else:
                            label = [label] * len(word)
                            for i in range(1, len(word)):
                                label[i] = "I{}".format(label[i][1:])
                        sample.extend(list(zip(word, label)))

    return sample_list


def collect_data(train_data_file, valid_data_file, save_path):
    data_file = [f for f in glob.glob(train_data_file + "/*.txt", recursive=True)]
    train_set = []
    for item in data_file:
        train_set.extend(parse_raw_vlsp2016(item))

    data_file = [f for f in glob.glob(valid_data_file + "/*.txt", recursive=True)]
    valid_set = []
    for item in data_file:
        valid_set.extend(parse_raw_vlsp2016(item))

    list_dataset = [valid_set, train_set]

    file_dev = open(os.path.join(save_path, 'valid.src'), 'w', encoding='utf-8')
    file_dev_label = open(os.path.join(save_path, 'valid.tgt'), 'w', encoding='utf-8')
    file_train = open(os.path.join(save_path, 'train.src'), 'w', encoding='utf-8')
    file_train_label = open(os.path.join(save_path, 'train.tgt'), 'w', encoding='utf-8')
    list_file = [(file_dev, file_dev_label), (file_train, file_train_label)]

    for dataset, (file_src, file_tgt) in zip(list_dataset, list_file):
        for (src, tgt) in tqdm(dataset, total=len(dataset), desc=str(file_src.name)):
            file_src.write('{}\n'.format(' '.join(src)))
            file_tgt.write('{}\n'.format(' '.join(tgt)))

    for item in list_file:
        item[0].close()
        item[1].close()


if __name__ == "__main__":
    import sys

    # collect_data('./data-bin/ner/raw/NER2016-TrainingData-3-3-2017-txt',
    #              './data-bin/ner/raw/Test data (16-9-2016) Column',
    #              './data-bin/ner/raw/')

    m_utils.import_user_module('./plugin')
    sys.argv += [
        '--source-lang', 'src',
        '--target-lang', 'tgt',
        '--task', 'normalize',
        '--trainpref', './data-bin/vlsp/raw/train',
        '--validpref', './data-bin/vlsp/raw/valid',
        '--destdir', 'data-bin-local/vlsp/preprocessed',
        # '--trained-lm', 'model-bin',
        '--nwordstgt', '100',
        '--nwordssrc', '10000',
        '--workers', '5',
    ]

    model_lm = XLMRModel.from_pretrained('/data/models/NLP/capu/capu-vi/model-bin/language_model/envibert')
    cli_main(pre_trained_lm=model_lm)
