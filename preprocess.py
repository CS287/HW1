#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re


def line_to_words(line, dataset):
    # Different preprocessing is used for these datasets.
    if dataset in ['SST1', 'SST2']:
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    words = clean_line.split(' ')
    words = words[1:]
    return words


def get_vocab(file_list, dataset=''):
    """
    Construct index feature dictionary.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    max_sent_len = 0
    word_to_idx = {}
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        if filename:
            with open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    words = line_to_words(line, dataset)
                    max_sent_len = max(max_sent_len, len(words))
                    for word in words:
                        if word not in word_to_idx:
                            word_to_idx[word] = idx
                            idx += 1
    return max_sent_len, word_to_idx


def convert_data(data_name, word_to_idx, max_sent_len, dataset, start_padding=0):
    """
    Convert data to padded word index features.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    features = []
    lbl = []
    with open(data_name, 'r', encoding="latin-1") as f:
        for line in f:
            words = line_to_words(line, dataset)
            y = int(line[0]) + 1
            sent = [word_to_idx[word] for word in words]
            sent = list(set(sent))
            # end padding
            if len(sent) < max_sent_len + start_padding:
                sent.extend([1] * (max_sent_len + start_padding - len(sent)))
            # start padding
            sent = [1]*start_padding + sent
            features.append(sent)
            lbl.append(y)
    return np.array(features, dtype=np.int32), np.array(lbl, dtype=np.int32)


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Different data sets to try.
# Note: TREC has no development set.
# And SUBJ and MPQA have no splits (must use cross-validation)
FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                       "data/stsa.fine.dev",
                       "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.phrases.train",
                       "data/stsa.binary.dev",
                       "data/stsa.binary.test"),
              "TREC": ("data/TREC.train.all", None,
                       "data/TREC.test.all"),
              "SUBJ": ("data/subj.all", None, None),
              "MPQA": ("data/mpqa.all", None, None)}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test = FILE_PATHS[dataset]

    # Features are just the words.
    max_sent_len, word_to_idx = get_vocab([train, valid, test], dataset)

    # Dataset name
    train_input, train_output = convert_data(train, word_to_idx, max_sent_len,
                                             dataset)
    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, max_sent_len,
                                                 dataset)
    if test:
        test_input, _ = convert_data(test, word_to_idx, max_sent_len,
                                 dataset)

    V = len(word_to_idx) + 1
    print('Vocab size:', V)

    C = np.max(train_output)

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
