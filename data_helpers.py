#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^(א-ת)A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(datasample_file_1, datasample_file_2, datasample_file_3, datasample_file_4, datasample_file_5):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    datasample_examples1 = list(open(datasample_file_1, "r").readlines())
    datasample_examples1 = [s.strip() for s in datasample_examples1]
    datasample_examples2 = list(open(datasample_file_2, "r").readlines())
    datasample_examples2 = [s.strip() for s in datasample_examples2]
    datasample_examples3 = list(open(datasample_file_3, "r").readlines())
    datasample_examples3 = [s.strip() for s in datasample_examples3]
    datasample_examples4 = list(open(datasample_file_4, "r").readlines())
    datasample_examples4 = [s.strip() for s in datasample_examples4]
    datasample_examples5 = list(open(datasample_file_5, "r").readlines())
    datasample_examples5 = [s.strip() for s in datasample_examples5]
    # Split by words
    x_text = datasample_examples1 + datasample_examples2 + datasample_examples3 + datasample_examples4 + datasample_examples5
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    datasample_labels1 = [[1, 0, 0, 0, 0] for _ in datasample_examples1]
    datasample_labels2 = [[0, 1, 0, 0, 0] for _ in datasample_examples2]
    datasample_labels3 = [[0, 0, 1, 0, 0] for _ in datasample_examples3]
    datasample_labels4 = [[0, 0, 0, 1, 0] for _ in datasample_examples4]
    datasample_labels5 = [[0, 0, 0, 0, 1] for _ in datasample_examples5]
    y = np.concatenate([datasample_labels1, datasample_labels2, datasample_labels3, datasample_labels4, datasample_labels5], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
