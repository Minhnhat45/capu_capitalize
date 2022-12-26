# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
    BaseFairseqModel,
)
from torchcrf import CRF
from fairseq.models.transformer import Embedding, TransformerDecoder, Linear
import torch
import math
import numpy as np
from fairseq.models.roberta import XLMRModel, RobertaHubInterface

DEFAULT_MAX_SOURCE_POSITIONS = 10240
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('bert_rnn_crf')
class BERTRNNModel(BaseFairseqModel):
    def __init__(self, encoder, rnn_transform, output_transform, tgt_dict, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.rnn_transform = rnn_transform
        self.crf = CRF(len(tgt_dict), batch_first=True)
        self.output_transform = output_transform
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src_tokens, src_lengths, src_subwords, src_words, target, **kwargs):
        encode_out = self.encoder.extract_features(src_tokens, **kwargs)[0]
        # src_lengths = torch.sum(src_tokens != 1, dim=-1)
        src_subwords = torch.tensor(src_subwords)
        src_words = torch.tensor(src_words)
        batch_size = src_subwords.shape[0]
        max_sub_word = src_subwords.shape[1]
        max_word = src_words.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))
        for i, sample_length in enumerate(src_words):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0
        if self.rnn_transform:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                encode_out, src_lengths.cpu(), enforce_sorted=False, batch_first=True
            )
            rnn_output, hidden = self.rnn_transform(packed)
            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )
            sentence_tensor = self.dropout(sentence_tensor)
            features = self.output_transform(sentence_tensor)
        else:
            encode_out = self.dropout(encode_out)
            features = self.output_transform(encode_out)
        #print("---src_words---")
        #for word in src_words:
        #    print(word)
        #print(features.size())
        features_cuda = features.to(align_matrix.dtype).cuda()
        align_matrix = align_matrix.to(features_cuda.device)
        features_convert = torch.bmm(align_matrix, features_cuda)
        #print("---Features_convert---")
        #print(features_convert.size())
        return features_convert

    def encode(self, src_tokens, src_lengths, src_subwords, src_words, **kwargs):
        encode_out = self.encoder.extract_features(src_tokens, **kwargs)[0]

        src_subwords = torch.tensor(src_subwords)
        src_words = torch.tensor(src_words)
        batch_size = src_subwords.shape[0]
        max_sub_word = src_subwords.shape[1]
        max_word = src_words.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))
        for i, sample_length in enumerate(src_words):
            #print(sample_length)
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0

        if self.rnn_transform:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                encode_out, src_lengths.cpu(), enforce_sorted=False, batch_first=True
            )
            rnn_output, hidden = self.rnn_transform(packed)
            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            features = self.output_transform(sentence_tensor)
        else:
            features = self.output_transform(encode_out)
        #features_cuda = torch.tensor(features, dtype=torch.float64).cuda()
        features_cuda = features.to(align_matrix.dtype).cuda()
        align_matrix = align_matrix.to(features_cuda.device)
        #print(align_matrix.size())
        #print(features_cuda)
        #align_tensor = torch.tensor(align_list).cuda()
        #print(align_tensor)
        features_convert = torch.bmm(align_matrix, features_cuda)

        mask_tensor = torch.arange(features_convert.size(1))[None, :].to(src_lengths.device) < src_lengths[:, None]
        #print(features_convert)
        return self.crf.decode(features_convert, mask=mask_tensor)
        #return features_convert
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.trained_lm:
            encoder_embed_tokens = task.trained_lm.model
        else:
            if args.share_all_embeddings:
                if src_dict != tgt_dict:
                    raise ValueError('--share-all-embeddings requires a joined dictionary')
                if args.encoder_embed_dim != args.decoder_embed_dim:
                    raise ValueError(
                        '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
                if args.decoder_embed_path and (
                        args.decoder_embed_path != args.encoder_embed_path):
                    raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
                encoder_embed_tokens = build_embedding(
                    src_dict, args.encoder_embed_dim, args.encoder_embed_path
                )
                # decoder_embed_tokens = encoder_embed_tokens
                args.share_decoder_input_output_embed = True
            else:
                encoder_embed_tokens = build_embedding(
                    src_dict, args.encoder_embed_dim, args.encoder_embed_path
                )
                # decoder_embed_tokens = build_embedding(
                #     tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
                # )

        # encoder = cls.build_encoder(args, tgt_dict, encoder_embed_tokens)
        if args.rnn_type in ['LSTM', 'GRU']:
            rnn_model = getattr(torch.nn, args.rnn_type)(
                task.trained_lm.args.encoder_embed_dim,
                task.trained_lm.args.encoder_embed_dim,
                num_layers=args.rnn_layers,
                dropout=0.0 if args.rnn_layers == 1 else args.dropout,
                bidirectional=True,
                batch_first=True)
            output_transform = torch.nn.Linear(
                # self.embeddings.embedding_length, len(tag_dictionary)
                task.trained_lm.args.encoder_embed_dim * 2, len(task.target_dictionary)
            )
        else:
            rnn_model = None
            output_transform = torch.nn.Linear(
                # self.embeddings.embedding_length, len(tag_dictionary)
                task.trained_lm.args.encoder_embed_dim, len(task.target_dictionary)
            )

        return BERTRNNModel(encoder_embed_tokens, rnn_model, output_transform, tgt_dict, dropout=args.dropout)

    # @classmethod
    # def build_encoder(cls, args, tgt_dict, embed_tokens):
    #     if args.trained_lm:
    #         xlmr_trained_class = XLMRModel.from_pretrained(args.trained_lm)
    #         language_model = xlmr_trained_class.model
    #         if not args.fine_tuning_lm:
    #             for param in language_model.parameters():
    #                 param.requires_grad = False
    #
    #         language_model.encoder_embed_dim = xlmr_trained_class.args.encoder_embed_dim
    #         return language_model
    #     else:
    #         return TransformerDecoderStandalone(args, tgt_dict, embed_tokens)

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.encoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class TransformerDecoderStandalone(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.project_in_combine_dim = Linear(args.input_dim, args.decoder_embed_dim, bias=False)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **unused)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
                Similar to *forward* but only return features.

                Returns:
                    tuple:
                        - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                        - a dictionary with any model-specific outputs
                """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if self.training:
            # Try dropout
            # self.dictionary.unk()
            size_raw = prev_output_tokens.size()
            prev_output_tokens = prev_output_tokens.view(-1)
            rand_index = torch.multinomial(prev_output_tokens.float(),
                                           math.floor(self.dropout * len(prev_output_tokens)))
            if rand_index.size(0) > 0:
                prev_output_tokens[rand_index] = self.dictionary.unk()
                prev_output_tokens = prev_output_tokens.view(size_raw)
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        # batch_size, sentence_length, word_length = unused['src_char_tokens'].size()
        # x_char = self.char_model(unused['src_char_tokens'].view(-1, word_length), None).view(batch_size,
        #                                                                                      sentence_length,
        #                                                                                      -1)

        # x = torch.cat([x, x_char], dim=-1)
        if self.project_in_combine_dim is not None:
            x = self.project_in_combine_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                x,
                None,
                incremental_state,
                None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}


@register_model_architecture('bert_rnn_crf', 'bert_rnn_crf')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.2)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.char_embed_dim = getattr(args, 'char_embed_dim', 256)
    args.char_hidden_dim = getattr(args, 'char_hidden_dim', 512)
    args.input_dim = getattr(args, 'input_dim', args.char_hidden_dim + args.decoder_embed_dim)
    args.word_max_length = getattr(args, 'word_max_length', 10)
    args.max_source_positions = getattr(args, 'max_source_positions', 10240)
