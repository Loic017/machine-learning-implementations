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

        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(
            2 / (self.input_size + self.output_size)
        )
        self.bias = np.zeros((1, output_size))

    def __repr__(self):
        return f"Linear Layer (in {self.input_size}, out {self.output_size}), (activation {self.activation})"

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights)
        # print(self.z.shape)
        self.z = self.z + self.bias
        # print(self.z.shape)
        self.a = self.activation.forward(x=self.z)
        return self.a

    def backward(self, output_grad):
        """
        Performs backpropagation on the layer.

        Args:
            output_grad (np.ndarray): grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            next_grad (np.ndarray): Gradient of the loss with respect to the layer's input.
            weights_grad (np.ndarray): Gradient of the loss with respect to the weights.
            bias_grad (np.ndarray): Gradient of the loss with respect to the bias.
        """

        # Step 1: Backprop through the activation function.
        # Compute the derivative of the activation function at the pre-activation values.
        activation_derivative = self.activation.backward(self.z)

        # Multiply element-wise with the gradient coming from the next layer.
        grad_pre_activation = output_grad * activation_derivative

        # Step 2: Compute gradients with respect to weights and biases.
        # The gradient for the weights is computed as the dot product of the transpose of the input
        # and the gradient from the pre-activation, averaged over the batch.

        weights_grad = np.dot(self.input.T, grad_pre_activation)
        # The gradient for the bias is the average of the gradients from the pre-activation over the batch.
        bias_grad = np.sum(grad_pre_activation, axis=0, keepdims=True)

        # Step 3: Compute the gradient to propagate to the previous layer.
        # This is the dot product of the gradient from the pre-activation and the transpose of the weights.
        next_grad = np.dot(grad_pre_activation, self.weights.T)

        return next_grad, weights_grad, bias_grad

    def update(self, lr, weights_grad, bias_grad):
        # weights_grad = np.mean(weights_grad, axis=1)
        # bias_grad = np.mean(bias_grad, axis=1)

        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1).reshape(-1, 1)

    linear = Linear(1, 1, af.Sigmoid())
    print(linear.forward(x))
    print(linear)
