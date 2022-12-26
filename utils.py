import os
import sys
import importlib
import re
from fairseq import utils as fairseq_utils, checkpoint_utils
import argparse

SPACE_NORMALIZER = re.compile(r"\s+")


def import_user_module(module_path):
    if module_path is not None:
        module_path = os.path.abspath(module_path)
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


def tokenize_line_word(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_line_char(line):
    line = SPACE_NORMALIZER.sub("", line)
    line = line.strip()
    return list(line)


def shell_sort(arr):
    # Start with a big gap, then reduce the gap
    n = len(arr)
    gap = n // 2

    # Do a gapped insertion sort for this gap size.
    # The first gap elements a[0..gap-1] are already in gapped
    # order keep adding one more element until the entire array
    # is gap sorted
    while gap > 0:

        for i in range(gap, n):

            # add a[i] to the elements that have been gap sorted
            # save a[i] in temp and make a hole at position i
            temp = arr[i]

            # shift earlier gap-sorted elements up until the correct
            # location for a[i] is found
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap

                # put temp (the original a[i]) in its correct location
            arr[j] = temp
        gap //= 2


def load_model_ensemble_and_task(filenames, arg_overrides=None, task=None):
    from fairseq import tasks

    ensemble = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))
        state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)

        args = state["args"]
        if task is None:
            task = tasks.setup_task(args)

        # build model for ensemble
        model = task.build_model(args)
        model.load_state_dict(state["model"], strict=False, args=args)
        ensemble.append(model)
    return ensemble, args, task


def from_pretrained(
        model_name_or_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='.',
        archive_map=None,
        **kwargs
):
    from fairseq import file_utils

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

        # allow archive_map to set default arg_overrides (e.g., tokenizer, bpe)
        # for each model
        if isinstance(model_name_or_path, dict):
            for k, v in model_name_or_path.items():
                if k == 'checkpoint_file':
                    checkpoint_file = v
                elif (
                        k != 'path'
                        # only set kwargs that don't already have overrides
                        and k not in kwargs
                ):
                    kwargs[k] = v
            model_name_or_path = model_name_or_path['path']

    model_path = file_utils.load_archive_file(model_name_or_path)

    # convenience hack for loading data and BPE codes from model archive
    if data_name_or_path.startswith('.'):
        kwargs['data'] = os.path.abspath(os.path.join(model_path, data_name_or_path))
    else:
        kwargs['data'] = file_utils.load_archive_file(data_name_or_path)
    for file, arg in {
        'code': 'bpe_codes',
        'bpecodes': 'bpe_codes',
        'sentencepiece.bpe.model': 'sentencepiece_vocab',
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            kwargs[arg] = path

    if 'user_dir' in kwargs:
        fairseq_utils.import_user_module(argparse.Namespace(user_dir=kwargs['user_dir']))

    models, args, task = load_model_ensemble_and_task(
        [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(':')],
        arg_overrides=kwargs,
    )

    return {
        'args': args,
        'task': task,
        'models': models,
    }


def load_pretrain_lm(model_path):
    from fairseq.models.roberta import RobertaHubInterface, XLMRModel
    x = from_pretrained(
        model_path,
        checkpoint_file='model.pt', data_name_or_path='.',
        archive_map=XLMRModel.hub_models(),
        bpe='sentencepiece',
        load_checkpoint_heads=True
    )
    trained_lm: XLMRModel = RobertaHubInterface(x['args'], x['task'], x['models'][0])
    # src_dict = trained_lm.task.source_dictionary
    return trained_lm


def clean_state_dict(model_path_target):
    import torch
    norm_model_path = './model-bin/entity_recognition/checkpoint_best.pt'
    # lm_model_path = './model-bin/language_model/xlmr.base/model.pt'
    state = torch.load(norm_model_path, map_location="cpu")
    # lm_state = torch.load(lm_model_path, map_location="cpu")
    del state['last_optimizer_state']
    # lm_state['model'] = {}
    torch.save(state, os.path.join(model_path_target, 'er_model.pt'))
    # torch.save(lm_state, os.path.join(model_path_target, 'model.pt'))


if __name__ == "__main__":
    clean_state_dict('./model-bin/entity_recognition_infer')
    # # load_pretrain_lm('./model-bin/language_model/vibert')
    #
    # import sentencepiece as spm
    #
    # sp = spm.SentencePieceProcessor()
    # sp.Load('./model-bin/language_model/vibert/sentencepiece.bpe.model')
    #
    # print(sp.encode('Gần ngôi làng Nazca của người Peru'))
