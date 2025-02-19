import numpy as np
import matplotlib.pyplot as plt
import loss_functions as ls
import activation_functions as af


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

        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def __repr__(self):
        return f"Linear Layer (in {self.input_size}, out {self.output_size}), (weights {self.weights}, bias {self.bias}), (activation {self.activation})"

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.a = self.activation.forward(x=self.z)
        return self.a

    def backward(self, output_grad):
        """
        Performs backpropagation on the layer.

        Args:
            output_grad (np.ndarray): The gradient of the loss with respect to the output of the layer. For the last layer, this is the gradient of the loss with respect to the output of the model.

        Returns:
            next_grad (np.ndarray): The gradient of the loss with respect to the input of the layer. For the first layer, this is the gradient of the loss with respect to the input of the model.
            weights_grad (np.ndarray): The gradient of the loss with respect to the weights of the layer.
            bias_grad (np.ndarray): The gradient of the loss with respect to the bias of the layer.
        """
        dprior_z_dcurr_output = output_grad
        dcurr_output_dcurr_z = self.activation.backward(self.z)

        next_grad = dprior_z_dcurr_output * dcurr_output_dcurr_z
        weights_grad = np.dot(self.input.T, next_grad) / self.input.shape[0]
        bias_grad = np.sum(next_grad, axis=0, keepdims=True) / self.input.shape[0]

        # print(f"output_grad shape {output_grad.shape}")
        # print(f"next_grad shape {next_grad.shape}")
        # print(f"weights_grad shape {weights_grad.shape}")
        # print(f"bias_grad shape {bias_grad.shape}")

        return next_grad, weights_grad, bias_grad

    def update(self, lr, weights_grad, bias_grad):
        weights_grad = np.mean(weights_grad, axis=1)
        bias_grad = np.mean(bias_grad, axis=1)
        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1).reshape(-1, 1)

    linear = Linear(1, 1, af.Sigmoid())
    print(linear.forward(x))
    print(linear)
