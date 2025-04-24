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

    def backward(self, grad=1):
        return grad * (1 - self.output**2)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "relu"

    def forward(self, x):
        self.x = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad=1):
        return grad * (self.x > 0)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __repr__(self):
        return "leaky_relu"

    def forward(self, x):
        self.x = x
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output

    def backward(self, grad=1):
        return grad * np.where(self.x > 0, 1, self.alpha)


# https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/#softmax
# https://www.youtube.com/watch?v=M59JElEPgIg
# https://www.youtube.com/watch?v=KpKog-L9veg&t=550s&pp=ygUbc29mdG1heCBhY3RpdmF0aW9uIGZ1bmN0aW9u
# class SoftMax(Activation):
#     def __init__(self):
#         super().__init__()

#     def __repr__(self):
#         return "softmax"

#     def forward(self, x):
#         x_stable = x - np.max(x, axis=-1, keepdims=True)
#         exp = np.exp(x_stable)
#         self.output = exp / np.sum(exp, axis=-1, keepdims=True)
#         return self.output

#     def backward(self, grad=1):
#         # https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
#         # https://github.com/eliben/deep-learning-samples/blob/main/softmax/softmax.py

#         SM = self.output.reshape((-1, 1))
#         jac = np.diagflat(self.output) - np.dot(SM, SM.T)
#         return grad


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

    # softmax = SoftMax()
    # softmax_forward = softmax.forward(x)
    # softmax_backward = softmax.backward()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(x, sigmoid_forward)
    axes[0, 0].set_title("Sigmoid")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Sigmoid( and derivative) and derivative")
    axes[0, 0].grid()
    axes[0, 0].plot(x, sigmoid_backward)

    axes[0, 1].plot(x, tanh_forward)
    axes[0, 1].set_title("Tanh")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Tanh(x) and derivative")
    axes[0, 1].grid()
    axes[0, 1].plot(x, tanh_backward)

    axes[1, 0].plot(x, relu_forward)
    axes[1, 0].set_title("ReLU")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("ReLU(x) and derivative")
    axes[1, 0].grid()
    axes[1, 0].plot(x, relu_grad)

    # axes[1, 1].plot(x, softmax_forward)
    # axes[1, 1].set_title("Softmax")
    # axes[1, 1].set_xlabel("x")
    # axes[1, 1].set_ylabel("Softmax(x)")
    # axes[1, 1].grid()
    # axes[1, 1].plot(x, softmax_backward)

    plt.tight_layout()
    plt.show()
