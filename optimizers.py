import numpy as np
import matplotlib.pyplot as plt


def central_difference_approximation(function, x, idx, eps):
    x_up = x[idx] + eps
    x_down = x[idx] - eps

    return (function(x_up) - function(x_down)) / (2 * eps)


def get_gradient(loss_function, params, eps=1e-5):
    gradients = np.zeros_like(params)
    for i, v in enumerate(params):
        gradients[i] = central_difference_approximation(loss_function, params, i, eps)

    return gradients


def update_parameter(parameter, lr, gradient):
    return (
        parameter - lr * gradient
    )  # Subtract to get the negative gradient (influenced by the learning rate)


def gradient_descent(params, loss_function, lr=0.01, eps=1e-5):
    gradients = get_gradient(loss_function, params, eps)
    updated_params = update_parameter(params, lr, gradients)
    return updated_params


if __name__ == "__main__":
    pass
