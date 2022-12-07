# utils

import numpy as np


def shuffle_dataset_and_label(dataset, label):
    data_l = list(zip(dataset, label))
    np.random.shuffle(data_l)
    dataset_, label_ = zip(*data_l)
    return np.array(dataset_), np.array(label_)
