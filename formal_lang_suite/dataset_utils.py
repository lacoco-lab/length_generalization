import os
import torch
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset, DataLoader
from hydra.core.hydra_config import HydraConfig

from dataloader import dump_data_to_file
from dataloader_utils import (
    create_corpus_non_star_free,
    create_corpus_tomita,
    create_corpus_star_free,
)


logger = logging.getLogger()


def tokenize_output_string(output_string, chunk_size) -> List[str]:
    # Tokenize the output string into chunks of size chunk_size
    tokenized = [output_string[i:i+chunk_size] for i in range(0, len(output_string), chunk_size)]
    # Find the unique chunks
    unique_tokens = list(set(tokenized))
    possible_chars = [chr(i) for i in range(len(unique_tokens) + 5)]
    # Remove the EOS, BOS, and PAD tokens from the possible characters
    if '$' in possible_chars:
        possible_chars.remove('$')
    if '#' in possible_chars:
        possible_chars.remove('#')
    if '.' in possible_chars:
        possible_chars.remove('.')
    token_to_letter = {unique_tokens[i]: possible_chars[i] for i in range(len(unique_tokens))}
    # Replace the chunks with the letters
    return [token_to_letter[token] for token in tokenized]


def add_special_tokens(s, max_length, bos_token='$', eos_token='#', pad_token='.'):
    # Add special tokens to the string and pad to the max length
    # Effective maximum length becomes maximum_length + 2 ; since the string with max tokens
    # will also be appended with an extra bos before and after.
    if isinstance(s, list):
        s = ''.join(s)
    return bos_token + s + eos_token + (max_length - len(s)) * pad_token


def preprocess_input_output(input_strings, output_strings, chunk_size, bos_token='$', eos_token='#', pad_token='.'):
    # Get the maximum length from both input and output strings
    max_length = max([len(i) for i in input_strings])
    if chunk_size > 1:
        # If the output > input ; then group the output chunks into the same alphabet 
        output_strings = list(map(lambda x: tokenize_output_string(x, chunk_size), output_strings))
    max_length = max(max_length, max([len(i) for i in output_strings]))
    
    # Add bos, eos, pad to each of the input and output strings as required.
    input_strings = list(map(lambda s: add_special_tokens(s, max_length, bos_token, eos_token, pad_token), input_strings))
    output_strings = list(map(lambda s: add_special_tokens(s, max_length, bos_token, eos_token, pad_token), output_strings))
    return input_strings, output_strings
    

def get_dataset_reqs(input_strings, output_strings):

    vocab_inp = set([j for i in input_strings for j in i])
    vocab_out = set([j for i in output_strings for j in i])

    vocab = vocab_inp.union(vocab_out)

    vocab = pd.get_dummies(list(vocab))

    # Convert character to index
    encoder = vocab.idxmax()
    decoder = vocab.idxmax(axis=1).to_dict()
    return vocab, encoder, decoder


class DatasetClass(Dataset):
    def __init__(self, input_strings, output_strings, vocab, encoder, decoder):
        self.input_strings = input_strings
        self.output_strings = output_strings
        # Write the variables to the class for easier access
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder

    def __len__(self) -> int:
        return len(self.input_strings)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:

        return {
            "input": DatasetClass.str_to_tensor_index_input(self.input_strings[idx], self.encoder, torch.int),
            "output": DatasetClass.str_to_tensor_index_input(self.output_strings[idx], self.encoder, torch.int)
        }

    @classmethod
    def str_to_tensor_index_input(cls, input, encoder, dtype=torch.float) -> torch.Tensor:
        # Converts to tensor after converting sequence of chars to indices from the vocabulary
        # + 1, since 0 is reserved for the padding index
        return torch.tensor([encoder[i] + 1 for i in input], dtype=dtype)


def read_file(file_path) -> List[str]:
    with open(file_path) as f:
        data = f.readlines()
        data = [d.replace('\n', '') for d in data]
    return data


def build_lang_config(dataset_config):
    """
    Build language generator config from hydra config
    Args:
        dataset_config: hydra dataset config

    Returns:
        lang_config: dict of language parameters

    """
    lang_config = {}
    for param, value in dataset_config.items():
        lang_config[param] = value

    return lang_config


def dump_datasets_locally(data_dir, train_dataset, all_val_datasets, data_type):
    # data_type : src / tgt
    dict_key = 'train_' + data_type
    dumped_data_list = [{dict_key: train_dataset}]
    location_file = dict_key + '.txt'
    dump_data_to_file(data_dir, location_file, train_dataset)

    for idx, val_dataset in enumerate(all_val_datasets):
        # Note the validation bin number
        key_str = 'val_' + data_type + '_bin' + str(idx)
        # Write to the file
        dump_data_to_file(data_dir, key_str + ".txt", val_dataset)
        # Append to the list, to be used later for creating dataloaders and datasets
        dumped_data_list.append({key_str: val_dataset})
    return dumped_data_list


def exists_dataset(data_dir):
    files = ['train_src.txt', 'train_tgt.txt']
    for file in files:
        path = str(data_dir) + '/' + file
        if os.path.exists(path) is False:
            return False
    
    print("Data generation aborted. Datasets already exist.")
    return True


# -> Union[Union[Dict, MultipleLenDataset], Union[Dict, SameLenDataset]]:
def create_dataloader(base_folder, batch_size, lang_params=None, bos_token='$', eos_token='#',
                      pad_token='.', max_size=0, num_val_bins=1, generate=False):

    dataloader_dict = {}
    dataset_name = base_folder
    base_folder = Path(HydraConfig.get().runtime.cwd) / 'generated_ds' / dataset_name
    chunk_size = lang_params.get('chunk_size', 1)
    curr_lang_fam = lang_params['lang_fam']
    max_sequence_leng = 0

    # Generate the dataset, if it doesn't exist already
    if generate and not exists_dataset(base_folder):
        if curr_lang_fam == 'Tomita':
            train_corpus, val_corpus_bins = create_corpus_tomita(lang_params)
        elif curr_lang_fam == 'NonStarFree':
            train_corpus, val_corpus_bins = create_corpus_non_star_free(lang_params)
        elif curr_lang_fam == 'StarFree':
            train_corpus, val_corpus_bins = create_corpus_star_free(lang_params)

        # Get directory to save generated datasets to
        data_dir = Path(HydraConfig.get().runtime.cwd) / 'generated_ds' / base_folder

        # Join training and validation data together
        problem_data = dump_datasets_locally(data_dir, train_corpus.source, [v.source for v in val_corpus_bins], 'src')
        solution_data = dump_datasets_locally(data_dir, train_corpus.target, [v.target for v in val_corpus_bins], 'tgt')

    # Common pipeline, for reusing data that has been generated already, or for newly generated data
    problem_files = {'train': base_folder / "train_src.txt"}
    solution_files = {'train': base_folder / "train_tgt.txt"}

    # Add the given number of validation files, and create dataloaders out of them as well
    for bin_number in range(num_val_bins):
        bin_number = str(bin_number)
        key_str = 'val_bin' + bin_number
        problem_files[key_str] = base_folder / ('val_src_bin' + bin_number + '.txt')
        solution_files[key_str] = base_folder / ('val_tgt_bin' + bin_number + '.txt')

    # Read in the input & output for the given problem at hand
    problem_data_dict = {category: read_file(f) for category, f in problem_files.items()}
    solution_n_data_dict = {category: read_file(f) for category, f in solution_files.items()}

    input_list, output_list = [], []
    category_data = {}
    for category, problem_data in problem_data_dict.items():
        if max_size > 0:
            # Restrict the length of sequences to be only the given size (For debugging)
            problem_data = [p[:max_size] for p in problem_data]
            solution_data = [p[:max_size] for p in solution_n_data_dict[category]]
        else:
            solution_data = solution_n_data_dict[category]

        input_strings, output_strings = preprocess_input_output(problem_data, solution_data, chunk_size, bos_token, eos_token, pad_token)
        input_list.extend(input_strings)
        output_list.extend(output_strings)
        category_data[category] = {'input': input_strings, 'output': output_strings}
            
    vocab, encoder, decoder = get_dataset_reqs(input_list, output_list)
    for category, data_dict in category_data.items():
        
        dataset = DatasetClass(data_dict['input'], data_dict['output'], vocab, encoder, decoder)
        dataloader_dict[category] = DataLoader(dataset, batch_size=batch_size)
        # The input length = output length, hence redundant to check both
        max_sequence_length = max(max_sequence_leng, max([len(i) for i in data_dict['input']]))

    return dataloader_dict, dataset, max_sequence_length
