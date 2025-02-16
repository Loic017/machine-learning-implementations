import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)

    plt.plot(x, y_sigmoid, label="sigmoid")
    plt.plot(x, y_tanh, label="tanh")
    plt.plot(x, y_relu, label="relu")
    plt.legend()
    plt.show()
