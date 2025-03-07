import numpy as np
import matplotlib.pyplot as plt


# MSE
def mse(y_true, y_hat, grad=False):
    if grad:
        return -2 * (y_true - y_hat)
    return np.mean((y_true - y_hat) ** 2)


# Cross-entropy
# https://www.youtube.com/watch?v=6ArSys5qHAU
# https://www.youtube.com/watch?v=xBEh66V9gZo
# https://github.com/xbeat/Machine-Learning/blob/main/Cross-Entropy%20in%20Python.md
# https://www.youtube.com/watch?v=Pwgpl9mKars
# https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/#softmax


# def binary_cross_entropy(y_true, y_hat, grad=False):
#     epsilon = 1e-12
#     y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
#     if grad:
#         return (y_hat - y_true) / (y_hat * (1 - y_hat))
#     return (-1 / np.size(y_true)) * np.sum(
#         y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
#     )


# def multi_cross_entropy(y_true, y_hat, grad=False):
#     epsilon = 1e-15
#     y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
#     if grad:
#         return y_true - y_hat
#     return np.mean(-np.sum(y_true * np.log(y_hat), axis=1))
