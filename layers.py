import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self):
        pass

    def __repr__(self):
        return NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self, lr):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = np.random.randn(input_size * output_size)
        self.bias = np.zeros((1, input_size))

    def __repr__(self):
        return f"Linear Layer (in {self.input_size}, out {self.output_size})"

    def forward(self, x):
        return self.activation(np.dot(self.weights, x) + self.bias)
