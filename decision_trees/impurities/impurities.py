import numpy as np


def gini(samples_in_node: np.ndarray):
    probs = samples_in_node / np.sum(samples_in_node)
    return 1 - np.sum(probs**2)


# def entropy():
#     return NotImplementedError
