import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset, LanguagePairDataset


def collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
        input_feeding=True,
):
    if len(samples) == 0:
        return {}
    pad_idx = 0
    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # src_char_tokens = torch.cat([s['source_char'].unsqueeze(0) for s in samples], 0)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    #print("----src_lengths----")
    #print(src_lengths)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    #print("----src_tokens----")
    #print(src_tokens)
    src_words = []
    src_subwords = []
    #def merge_src(src):
    #   size = max(len(s) for s in src)
    #   for sen in src:
    #       if len(sen) < size:
    #           for _ in range(size - len(sen)):
    #               sen.extend([0])
    #   return src
    #src_words = []
    #src_subwords = []
    #for s in samples:
    #    src_words.append(s['word_encode'].numpy().tolist())
    #    src_subwords.append(s['source_word_size'].numpy().tolist())
    #src_words = merge_src(src_words)
    #src_words = torch.tensor(src_words)
    #src_words = src_words.index_select(0, sort_order)
    #src_words = src_words.numpy().tolist()
    #
    #src_subwords = merge_src(src_subwords)
    #src_subwords = torch.tensor(src_subwords)
    #src_subwords = src_subwords.index_select(0, sort_order)
    #src_subwords = src_subwords.numpy().tolist()

    src_subwords = merge('source_word_size', left_pad=left_pad_source)
    src_subwords = src_subwords.index_select(0, sort_order)
    src_subwords = src_subwords.numpy().tolist()

    src_words = merge('word_encode', left_pad=left_pad_source)
    src_words = src_words.index_select(0, sort_order)
    src_words = src_words.numpy().tolist()
    
    #print("----src_words_lengths----")
    #print(src_words_lengths)
    #print("----src_words----")
    #print(src_words)
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    #print("--------------Matrix---------------")
    #for i, sample in enumerate(samples):
    #    print("This is the samples number: " + str(i))
    #    print(sample)
    #align_matrix = samples[1].get('align_matrix')
    #print(target.size())
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_subwords': src_subwords,
            'src_words': src_words,
            #'src_words_lengths': src_words_lengths,
            'target': target,
        },
        'target': target,
    }
    return batch


class NormalizeDataset(LanguagePairDataset):

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        #src_item = torch.tensor(src_item.numpy()[:-2])
        
        word_length = src_item[-2]
        subword_length = src_item[-1]
        
       
        word_encode = src_item[int(subword_length) * 2: -2]
        subword_encode = src_item[int(subword_length): -int(len(word_encode) + 2)]
        src_item = torch.tensor(src_item[:-int(len(word_encode) + 2)])
        
        #src_item = torch.tensor(src_item[:(int(subword_length) * 2)])
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        #print(tgt_item)
        #src_item = torch.tensor(src_item.numpy()[:-2])
        src_item = src_item.reshape((2, -1))
        src_subword = src_item[0]
        src_word_size = src_item[1]
        #print(src_word_size)
        #print(word_encode)
        #tgt_norm = torch.zeros(len(word_encode), dtype=torch.int64)
        #tgt_norm = torch.zeros(len(src_word_size), dtype=torch.int64)
        #tgt_norm.fill_(self.tgt_dict.pad_index)

        #tgt_norm[0] = self.tgt_dict.bos_index
        #tgt_norm[-1] = self.tgt_dict.eos_index
        ##tgt_norm[1:-1].scatter_(0, torch.where(word_encode[1:-1] > 0)[0], tgt_item)
        #tgt_norm[1:-1].scatter_(0, torch.where(src_word_size[1:-1] > 0)[0], tgt_item)
        tgt_norm = [0]
        tgt_norm.extend(tgt_item.numpy().tolist())
        tgt_norm.append(2)
        tgt_norm = torch.tensor(tgt_norm)
        #print(tgt_norm)
        assert torch.sum(tgt_norm != self.tgt_dict.pad_index) - 2 == torch.sum(tgt_item > 0)

        # if src_item.size(0) != tgt_item.size(0) and src_item.size(0) % tgt_item.size(0) == 0:
        #     src_words = src_item[:tgt_item.size(0)]
        #     src_chars = src_item[tgt_item.size(0):].view(tgt_item.size(0), -1)
        # else:
        #     src_words = src_item
        #     src_chars = None

        return {
            'id': index,
            'source': src_subword,
            'source_word_size': src_word_size,
            'word_length': word_length,
            #'align_matrix': align_matrix,
            'word_encode': word_encode,
            'subword_encode': subword_encode,
            'target': tgt_norm,
        }

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
