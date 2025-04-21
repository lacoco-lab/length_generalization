from generators import tomita_generator
from generators import starfree_generator
from generators import nonstarfree_generator
from generators.starfree_generator import AB_D_BC, ZOT_Z_T
from typing import List, Tuple, Dict


class TomitaCorpus(object):
    def __init__(self, n, lower_window, upper_window, size, unique, leak=False, debug=False):
        assert n > 0 and n <= 7
        L = (lower_window + upper_window) // 2
        p = L / (2 * (1 + L))
        self.unique = unique
        self.leak = leak
        self.Lang = getattr(tomita_generator, 'Tomita{}Language'.format(n))(p, p)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size, lower_window, upper_window):
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window, self.leak)

        if self.unique:
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            inputs = list(inputs)
            outputs = list(outputs)

        return inputs, outputs


class StarFreeCorpus(object):

    def __init__(self, lang, num_par: int, lower_window: int, upper_window: int, size: int, unique: bool = False):
        # num_par -> refers to the depth, eg: in case of D languages, D2, D4, etc. 
        self.Lang = getattr(starfree_generator, lang+'Language')(num_par)
        # Remove duplicate entries from the dataset
        self.unique = unique
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int)-> Tuple[List[str], List[str]]:
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
        if self.unique:
            # Remove duplicate entries from the datasets
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            inputs, outputs = list(inputs), list(outputs)
        return inputs, outputs


class StarFreePostLanguageCorpus(object):
    def __init__(self, mandatory: str, pre_choices: str, post_choices: str, lower_window: int, upper_window: int, size: int):        
        if mandatory == 'd':
            self.lang = AB_D_BC(pre_choices, post_choices, mandatory)
        elif mandatory == '0':
            self.lang = ZOT_Z_T(pre_choices, post_choices, mandatory)
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int)-> Tuple[List[str], List[str]]:
        inputs, outputs = self.lang.training_set_generator(size, lower_window, upper_window)
        # Remove duplicate entries from the datasets
        inputs, outputs = zip(*set(zip(inputs, outputs)))
        return list(inputs), list(outputs)


class NonStarFreeCorpus(object):
    def __init__(self, lang, num_par: int, lower_window: int, upper_window: int, size: int, unique: bool = False):
        # num_par : number of repititions in case of AAStar, AA: 2, AAAA: 4 | for ABAB, its number of characters
        self.Lang = getattr(nonstarfree_generator, lang + 'Language')(num_par)
        # To remove duplicate entries from the dataset
        self.unique = unique
        self.source, self.target = self.generate_data(size, lower_window, upper_window)

    def generate_data(self, size: int, lower_window: int, upper_window: int) -> Tuple[List[str], List[str]]:
        inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
        if self.unique:
            # Remove duplicate entries from the dataset
            inputs, outputs = zip(*set(zip(inputs, outputs)))
            inputs, outputs = list(inputs), list(outputs)
        return inputs, outputs


def dump_data_to_file(data_dir, file_name, corpus):
    """
    Dump corpus to train and target files.
    Args:
        data_dir: name of the directory to save the corpus
        file_name: name of the file
        corpus: train or validation corpus generated
    """
    path = str(data_dir) + "/" + file_name
    with open(path, 'w') as file:
        file.write('\n'.join(corpus))
