import numpy as np
import matplotlib.pyplot as plt


# MSE
def mse(y_true, y_hat, grad=False):
    if grad:
        batch_size = y_hat.shape[0]
        return (2 / batch_size) * (y_hat - y_true)
    return np.mean((y_hat - y_true) ** 2)


# Cross-entropy
# https://www.youtube.com/watch?v=6ArSys5qHAU
# https://www.youtube.com/watch?v=xBEh66V9gZo
# https://github.com/xbeat/Machine-Learning/blob/main/Cross-Entropy%20in%20Python.md
# https://www.youtube.com/watch?v=Pwgpl9mKars
# https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/#softmax


def binary_cross_entropy(y_true, y_hat, grad=False):
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    if grad:
        return (y_hat - y_true) / (y_hat * (1 - y_hat))
    return (-1 / np.size(y_true)) * np.sum(
        y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)
    )


# Softmax(xi​)=∑j​exp(xj​)exp(xi​)​
def softmax(x):
    # print("x", x)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def multi_cross_entropy(y_true, y_hat, grad=False, reduce="mean", softmax_enable=True):
    epsilon = 1e-20
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    if softmax_enable:
        y_hat = softmax(y_hat)

    output = -np.sum(y_true * np.log(y_hat), axis=1)

    if grad:
        return y_hat - y_true

    if reduce == "mean":
        return np.mean(output)
    if reduce == "sum":
        return np.sum(output)
    if reduce is None:
        return output
