import numpy as np
import matplotlib.pyplot as plt


def mse(y_true, y_hat):
    return np.mean((y_true - y_hat) ** 2)


def binary_cross_entropy(y_true, y_hat):
    return (-1 / np.size(y_true)) * np.sum(
        y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
    )


def multi_cross_entropy(y_true, y_hat):
    return (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_hat))
