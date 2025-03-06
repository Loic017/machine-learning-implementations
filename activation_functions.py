import numpy as np
import matplotlib.pyplot as plt


class Activation:
    def __init__(self):
        self.output = None

    def __repr__(self):
        return NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "sigmoid"

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad=1):
        return grad * (self.output) * (1 - self.output)


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "tanh"

    def forward(self, x):
        # self.output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1 - self.output**2)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "relu"

    def forward(self, x):
        # print(f"Input to ReLU forward: {x}")  # Debugging statement
        self.output = np.maximum(0, x)
        # print(f"Output from ReLU forward: {self.output}")  # Debugging statement
        return self.output

    def backward(self, grad):
        return grad * np.where(self.output > 0, 1, 0)


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)

    sigmoid = Sigmoid()
    sigmoid_forward = sigmoid.forward(x)
    sigmoid_backward = sigmoid.backward()

    tanh = Tanh()
    tanh_forward = tanh.forward(x)
    tanh_backward = tanh.backward()

    relu = ReLU()
    relu_forward = relu.forward(x)
    relu_grad = relu.backward()

    plt.plot(x, relu_forward, label="relu")
    plt.plot(x, relu_grad, label="relu")
    plt.legend()
    plt.show()
