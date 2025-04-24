import numpy as np
import matplotlib.pyplot as plt
import loss_functions as ls
import activation_functions as af

from utils import assert_shape


class Layer:  # from abc import ABC
    def __init__(self):
        pass

    def __repr__(self):
        return NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self, lr, grad):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(
            2 / self.input_size
        )
        self.bias = np.zeros((1, output_size))

    def __repr__(self):
        return f"Linear Layer (in {self.input_size}, out {self.output_size}), (activation {self.activation})"

    def forward(self, x):
        """
        Performs forward pass on the layer.
        """
        batch_size = x.shape[0]
        self.input = x
        # print(f"Weights shape {self.weights.shape}")
        # print(f"x shape {x.shape}")
        assert_shape(arr=x, expected_shape=(batch_size, self.input_size))

        self.z = np.dot(x, self.weights) + self.bias
        assert_shape(arr=self.z, expected_shape=(batch_size, self.output_size))

        if self.activation is not None:
            self.a = self.activation.forward(x=self.z)
            assert_shape(arr=self.a, expected_shape=(batch_size, self.output_size))
            return self.a
        return self.z

    def backward(self, prior_layer_grad):
        """
        Performs backpropagation on the layer.

        Args:
            prior_layer_grad (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            next_grad (np.ndarray): Gradient of the loss with respect to the layer's input.
            weights_grad (np.ndarray): Gradient of the loss with respect to the weights.
            bias_grad (np.ndarray): Gradient of the loss with respect to the bias.
        """

        # print(self.__repr__())

        # Step 1: (gradient of activation with respect to z) * (gradient of loss with respect to activation)
        # The multiplication is done within the backward method of the activation function
        if self.activation is None:
            grad_activation = prior_layer_grad
        else:
            grad_activation = self.activation.backward(prior_layer_grad)

        # print("grad_activation ", grad_activation.shape)
        # print("self.input ", self.input.shape, " self.input.T ", self.input.T.shape)

        # Step 2: (gradient of z with respect to weights) * (gradient of loss with respect to z)
        weights_grad = np.dot(self.input.T, grad_activation)

        # Step 3: (gradient of z with respect to bias) * (gradient of loss with respect to z)
        bias_grad = np.sum(grad_activation, axis=0)

        # Step 4: Propagate gradient to the previous layer
        next_grad = np.dot(grad_activation, self.weights.T)

        return next_grad, weights_grad, bias_grad

    def update(self, lr, weights_grad, bias_grad):
        # Average the gradients over the batch size
        weights_grad /= self.input.shape[0]
        bias_grad /= self.input.shape[0]

        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad


class Conv2d(Layer):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        activation,
        stride=1,
        padding=0,
        groups=1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size_h, self.kernel_size_w = kernel_size
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.weights = np.random.randn(
            self.output_size,
            self.input_size / self.groups,
            self.kernel_size_h,
            self.kernel_size_w,
        ) * np.sqrt(2 / (self.input_size * self.kernel_size_h * self.kernel_size_w))

        self.bias = np.zeros((1, self.output_size, 1, 1))

    def __repr__(self):
        return "Conv2d Layer"

    def forward(self, x):
        return NotImplementedError

    def backward(self, prior_layer_grad):
        return NotImplementedError


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __repr__(self):
        return "MaxPool2d Layer"

    def forward(self, x):
        return NotImplementedError

    def backward(self, prior_layer_grad):
        return prior_layer_grad  # Is not learnable, so skip the backward pass


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Flatten Layer"

    def forward(self, x):
        return x.reshape(x.shape[0], -1)

    def backward(self, prior_layer_grad):
        return prior_layer_grad # Is not learnable, so skip the backward pass


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1).reshape(-1, 1)

    linear = Linear(1, 1, af.Sigmoid())
    print(linear.forward(x))
    print(linear)
