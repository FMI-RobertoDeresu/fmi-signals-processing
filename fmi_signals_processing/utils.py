import numpy as np


def get_normalized_uniform(size):
    a = np.random.uniform(0, 1, size)
    a = a/sum(a)
    return a


def normalize_array(a):
    shape = np.shape(a)
    last_dimension = list(shape)[-1]
    flatten = np.ndarray.flatten(a)

    for i in np.arange(0, len(flatten), last_dimension):
        flatten[i:i+last_dimension] = flatten[i:i+last_dimension] / sum(flatten[i:i+last_dimension])

    a = np.ndarray.reshape(flatten, shape)
    return a
