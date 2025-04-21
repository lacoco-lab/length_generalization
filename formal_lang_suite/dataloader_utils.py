import copy
from dataloader import (
    TomitaCorpus,
    NonStarFreeCorpus,
    StarFreeCorpus,
    StarFreePostLanguageCorpus
)


def create_corpus_tomita(params):
    # Retrieve all the relevant parameters
    debug, is_leak, num_params = params['debug'], params['leak'], params['num_par']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    if not is_leak:
        corpus = TomitaCorpus(num_params, lower_window, upper_window, train_size + test_size, unique=True, debug=debug)
        # Train and validation created together, hence separate them out first
        train_corpus = copy.deepcopy(corpus)
        train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]

        val_corpus = copy.deepcopy(corpus)
        val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
        val_corpus_bins = [val_corpus]

        # Prepare to make validation sets for greater window sizes
        lower_window = upper_window + 1
        upper_window = upper_window + len_incr

        for i in range(num_val_bins - 1):
            print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
            val_corpus_bin = TomitaCorpus(num_params, lower_window, upper_window, test_size, unique=True, debug=debug)
            val_corpus_bins.append(val_corpus_bin)
            lower_window = upper_window
            upper_window = upper_window + params['len_incr']

    else:
        train_corpus = TomitaCorpus(num_params, lower_window, upper_window, train_size,
                                    unique=False, leak=True, debug=params['debug'])
        val_corpus_bins = [TomitaCorpus(num_params, lower_window, upper_window, test_size, unique=True, leak=True, debug=debug)]

        lower_window = upper_window + 1
        upper_window = upper_window + len_incr

        for i in range(num_val_bins-1):
            val_corpus_bin = TomitaCorpus(num_params, lower_window, upper_window, test_size, unique=True, leak=True, debug=debug)
            val_corpus_bins.append(val_corpus_bin)
    
    return train_corpus, val_corpus_bins

    
def create_corpus_non_star_free(params):
    ### ABABStar, AAStar, AnStarA2 ###
    # Retrieve the relevant parameters
    curr_lang, num_params = params['lang_class'], params['num_par']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']

    # Create the training and 1st validation bin
    train_corpus = NonStarFreeCorpus(curr_lang, num_params, lower_window, upper_window, train_size)
    val_corpus_bins = [NonStarFreeCorpus(curr_lang, num_params, lower_window, upper_window, test_size, unique = True)]

    # Increment for subsequent bins
    lower_window = upper_window + 1
    upper_window = upper_window + len_incr

    # Create one less validation bin, since validation bin already made above    
    for i in range(num_val_bins - 1):
        val_corpus_bin = NonStarFreeCorpus(curr_lang, num_params, lower_window, upper_window, test_size, unique = True)
        val_corpus_bins.append(val_corpus_bin)

        lower_window = upper_window
        upper_window += len_incr
    return train_corpus, val_corpus_bins


def create_corpus_star_free(params):
    # Retrieve the relevant parameters
    curr_lang = params['lang_class']
    train_size, test_size, num_val_bins = params['training_size'], params['test_size'], params['num_val_bins']
    lower_window, upper_window, len_incr = params['lower_window'], params['upper_window'], params['len_incr']
    
    if curr_lang != 'StarFreeSpecial':
        num_params, keep_unique = params['num_par'], params['unique']
        # Create the corpus in one go for a validation and training set
        corpus = StarFreeCorpus(curr_lang, num_params, lower_window, upper_window, train_size + test_size, keep_unique)
        # separate the combined corpus into train and validation sets
        train_corpus = copy.deepcopy(corpus)
        train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]
        
        val_corpus = copy.deepcopy(corpus)
        val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
        val_corpus_bins = [val_corpus]

        # Prepare upper window for other validation bins [Lower window unchanged unlike other cases]
        lower_window = upper_window + 1
        upper_window = upper_window + len_incr
        
        # Create one less validation bin, since validation bin already made above
        for i in range(num_val_bins - 1):
            print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
            val_corpus_bin = StarFreeCorpus(curr_lang, num_params, lower_window, upper_window, test_size, keep_unique)
            val_corpus_bins.append(val_corpus_bin)
            lower_window = upper_window + 1
            upper_window = upper_window + len_incr
    else: 
        mandatory, pre_choices, post_choices = params['mandatory'], params['pre_choices'], params['post_choices']
        # Create the corpus in one go for a validation and training set
        corpus = StarFreePostLanguageCorpus(mandatory, pre_choices, post_choices, lower_window, upper_window, train_size + test_size)
        # separate the combined corpus into train and validation sets
        train_corpus = copy.deepcopy(corpus)
        train_corpus.source, train_corpus.target = corpus.source[:train_size], corpus.target[:train_size]
        
        val_corpus = copy.deepcopy(corpus)
        val_corpus.source, val_corpus.target = corpus.source[train_size:], corpus.target[train_size:]
        val_corpus_bins = [val_corpus]

        # Prepare upper window for other validation bins [Lower window unchanged unlike other cases]
        upper_window = upper_window + len_incr
        
        # Create one less validation bin, since validation bin already made above
        for i in range(num_val_bins - 1):
            print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
            val_corpus_bin = StarFreePostLanguageCorpus(mandatory, pre_choices, post_choices, lower_window, upper_window, test_size)
            val_corpus_bins.append(val_corpus_bin)
            upper_window = upper_window + len_incr

    return train_corpus, val_corpus_bins