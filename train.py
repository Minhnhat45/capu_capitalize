#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from fairseq_cli import train
import utils as postag_utils

if __name__ == '__main__':
    import sys

    postag_utils.import_user_module('./plugin')
    sys.argv += [
        './data-bin/vlsp/preprocessed/',
        '--user-dir', './plugin',
        '--task', 'normalize',
        '-a', 'bert_rnn_crf',
        '--optimizer', 'adam',
        '--lr', '0.0001',
        #'--batch-size', '256',
        '-s', 'src', '-t', 'tgt',
        '--dropout', '0.3',
        '--max-tokens', '60000',
        '--min-lr', '1e-09',
        '--lr-scheduler', 'inverse_sqrt',
        '--weight-decay', '0.0001',
        '--criterion', 'crf_loss',
        '--max-epoch', '30',
        '--warmup-updates', '500',
        '--warmup-init-lr', '1e-07',
        '--adam-betas', '(0.9,0.98)',
        '--max-source-positions', '10240',
        '--save-dir', 'model-bin/vlsp',
        #'--save-interval-updates', '1000',
        #'--keep-interval-updates', '5',
        #'--keep-last-epochs', '4',
        '--no-epoch-checkpoints',
        '--dataset-impl', 'mmap',
        '--num-workers', '2',
        '--trained-lm', 'model-bin/language_model/envibert',
        '--fine-tuning-lm', 'True',
        '--restore-file', 'model-bin/vlsp/checkpoint_best.pt',
        '--rnn-type', 'GRU',
        '--rnn-layers', '4',
        '--fix-batches-to-gpus',
        '--tensorboard-logdir', 'visualize',
        '--ddp-backend', 'no_c10d',
        '--find-unused-parameters',
        '--fp16',
        #'--fp16-init-scale', '4',
        #'--threshold-loss-scale', '1',
        #'--fp16-scale-window', '128',
        #'--memory-efficient-fp16',
        #'--ddp-backend=no_c10d',
    ]
    train.cli_main()
