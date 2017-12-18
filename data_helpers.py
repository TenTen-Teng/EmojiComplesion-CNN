import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r'\n', r'', string)
     # sentence = re.sub(r'[,.\'\";?\-\!]', r' ',sentence)
    string = re.sub(r'[0-9]', r'', string)
#    return string
    return string.strip().lower()
"""
def load_data_and_labels(positive_data_file, negative_data_file):
    
##    Loads MR polarity data from files, splits the data into words and generates labels.
##    Returns split sentences and labels.
    
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
        
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
"""

def load_data_and_labels(emoji1_data_file, emoji0_data_file, emoji2_data_file, emoji3_data_file, emoji4_data_file, emoji5_data_file, emoji6_data_file, emoji7_data_file, emoji8_data_file, emoji9_data_file):
#def load_data_and_labels(emoji1_data_file, emoji0_data_file, emoji2_data_file, emoji3_data_file, emoji4_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    emoji1_examples = list(open(emoji1_data_file, "r").readlines())
    emoji1_examples = [s.strip() for s in emoji1_examples]

    emoji0_examples = list(open(emoji0_data_file, "r").readlines())
    emoji0_examples = [s.strip() for s in emoji0_examples]

    emoji2_examples = list(open(emoji2_data_file, "r").readlines())
    emoji2_examples = [s.strip() for s in emoji2_examples]

    emoji3_examples = list(open(emoji3_data_file, "r").readlines())
    emoji3_examples = [s.strip() for s in emoji3_examples]

    emoji4_examples = list(open(emoji4_data_file, "r").readlines())
    emoji4_examples = [s.strip() for s in emoji4_examples]

    emoji5_examples = list(open(emoji5_data_file, "r").readlines())
    emoji5_examples = [s.strip() for s in emoji5_examples]

    emoji6_examples = list(open(emoji6_data_file, "r").readlines())
    emoji6_examples = [s.strip() for s in emoji6_examples]

    emoji7_examples = list(open(emoji7_data_file, "r").readlines())
    emoji7_examples = [s.strip() for s in emoji7_examples]

    emoji8_examples = list(open(emoji8_data_file, "r").readlines())
    emoji8_examples = [s.strip() for s in emoji8_examples]

    emoji9_examples = list(open(emoji9_data_file, "r").readlines())
    emoji9_examples = [s.strip() for s in emoji9_examples]

    # Split by words
    x_text = emoji1_examples + emoji0_examples + emoji2_examples + emoji3_examples + emoji4_examples + emoji5_examples + emoji6_examples + emoji7_examples + emoji8_examples + emoji9_examples
#    x_text = emoji1_examples + emoji0_examples + emoji2_examples + emoji3_examples + emoji4_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels

    """
    emoji0_labels = [[1, 0, 0, 0, 0] for _ in emoji0_examples]
    emoji1_labels = [[0, 1, 0, 0, 0] for _ in emoji1_examples]
    emoji2_labels = [[0, 0, 1, 0, 0] for _ in emoji2_examples]
    emoji3_labels = [[0, 0, 0, 1, 0] for _ in emoji3_examples]
    emoji4_labels = [[0, 0, 0, 0, 1] for _ in emoji4_examples]
    """
    emoji0_labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in emoji0_examples]
    emoji1_labels = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0] for _ in emoji1_examples]
    emoji2_labels = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] for _ in emoji2_examples]
    emoji3_labels = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for _ in emoji3_examples]
    emoji4_lables = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0] for _ in emoji4_examples]
    emoji5_lables = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0] for _ in emoji5_examples]
    emoji6_lables = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0] for _ in emoji6_examples]
    emoji7_lables = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0] for _ in emoji7_examples]
    emoji8_lables = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0] for _ in emoji8_examples]
    emoji9_lables = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1] for _ in emoji9_examples]

#    y = np.concatenate([emoji0_labels, emoji1_labels, emoji2_labels, emoji3_labels, emoji4_labels],0)
    y = np.concatenate([emoji0_labels, emoji1_labels, emoji2_labels, emoji3_labels, emoji4_lables, emoji5_lables, emoji6_lables, emoji7_lables, emoji8_lables, emoji9_lables], 0)
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
