from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification, GPT2Config, TrainingArguments, Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random
from copy import deepcopy
import string
import argparse
import itertools
import os
from collections import Counter
from typing import Optional
from easydict import EasyDict

def make_configs(search_space: dict[str, list]):
    all_configs = []
    for hyperparams in itertools.product(*search_space.values()):
        config = EasyDict(dict(zip(search_space.keys(), hyperparams)))
        all_configs.append(config)
    return all_configs


class NoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return 0

class NoPEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()


class customTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.special_token = "#"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_id = normal_tkn_num + 4
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id, self.special_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token, self.special_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # self.vocab = {"0": 0, "1": 1}
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab[self.special_token] = self.special_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def __call__(self, strings: list[str] | str, **kwargs):
        # this func is not used, since the data generator does not generate str
        # string is tokenized by white space
        if type(strings) == str:
            strings = [strings]
        ids = []
        strings = [s.split(" ") for s in strings]
        max_len = max(map(lambda x: len(x), strings))
        for s in strings:
            ids.append( list(map(lambda x: self.vocab[x], s)) + [self.pad_token_id] * (max_len-len(s)) )

        return {"input_ids": torch.LongTensor(ids)}

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)
    


class UniqueCopyDataset(IterableDataset):
    def __init__(self, tokenizer: customTokenizer, length_range: tuple[int, int], max_test_length: int, diff_ij: int):
        super().__init__()
        self.tokenizer = tokenizer 
        self.range_min, self.range_max = length_range
        self.range_min = max(diff_ij, self.range_min)
        self.max_test_length = max_test_length
        self.diff_ij = diff_ij  # t-1, t-2,... attention
        assert len(tokenizer) - 5 >= max_test_length - (diff_ij-1)
        assert (max_test_length >= self.range_max) or (max_test_length == -1)    # the pos emb is initialized based on max_test_length

    def determine_n_positions(self):
        adjusted_length = (self.max_test_length + 1) // self.diff_ij * self.diff_ij - 1
        input_seq = random.sample(range(len(self.tokenizer)-5), adjusted_length-(self.diff_ij-1))
        output_seq = input_seq[::self.diff_ij]
        self.n_positions = 3 + adjusted_length + len(output_seq)
        return self.n_positions

    def __iter__(self):
        while True:
            min_v = self.range_min // self.diff_ij * self.diff_ij + (self.diff_ij - 1)
            length = random.randint(min_v, self.range_max)     # length of string to be copied
            adjusted_length = (length + 1) // self.diff_ij * self.diff_ij - 1
            input_seq = random.sample(range(len(self.tokenizer)-5), adjusted_length-(self.diff_ij-1))
            instance = [self.tokenizer.bos_token_id]
            instance.extend([self.tokenizer.special_token_id] * (self.diff_ij-1))
            instance.extend(input_seq)
            instance.append(self.tokenizer.sep_token_id)

            output_seq = input_seq[::self.diff_ij]
            instance.extend(output_seq)
            instance.append(self.tokenizer.eos_token_id)

            label = deepcopy(instance)
            # setting some tokens to [pad] will make the loss on these tokens (as pred targets) be ignored
            label[:adjusted_length+2] = [self.tokenizer.pad_token_id,] * (adjusted_length+2)   # bos + ... + sep 
            
            if self.max_test_length != -1:
                offset = random.randint(0, self.n_positions - len(instance))
            else:
                offset = 0
            pos_ids = list(range(offset, len(instance)+offset))

            yield instance, pos_ids, label


class EvalDataset(Dataset):
    def __init__(self, d: IterableDataset, num_data: int) -> None:
        super().__init__()
        self.data = []
        for i, item in enumerate(d):
            if i >= num_data:
                break
            self.data.append(item)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class customCollator():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        input_ids, pos_ids, labels = tuple(zip(*examples))
        max_len = max(len(item) for item in input_ids)

        [item.extend([self.pad_id,] * (max_len - len(item))) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)
        [item.extend([self.pad_id,] * (max_len - len(item))) for item in labels]
        labels = torch.LongTensor(labels)
        labels[labels == self.pad_id] = -100
        [item.extend([item[-1],] * (max_len - len(item))) for item in pos_ids]
        pos_ids = torch.LongTensor(pos_ids)
        
        batch = {"input_ids": input_ids, "position_ids": pos_ids, "labels": labels}
        return batch
    

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    predictions = np.argmax(shift_logits, axis=-1)
    correct = np.all((predictions == shift_labels) | (shift_labels == -100), axis=1)
    return {"acc": correct.sum() / len(correct)}

