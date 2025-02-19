import numpy as np
import matplotlib.pyplot as plt
from loss_functions import mse
from layers import Linear
from activation_functions import ReLU


class Model:
    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def __repr__(self):
        return self.layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        weights_grads = []
        bias_grads = []
        prior_loss = loss_grad
        for layer in reversed(self.layers):
            next_grad, weights_grad, bias_grad = layer.backward(prior_loss)

            weights_grads.append(weights_grad)
            bias_grads.append(bias_grad)

            prior_loss = next_grad

        weights_grads.reverse()
        bias_grads.reverse()
        return weights_grads, bias_grads

    def update(self, lr, weights_grads, bias_grads):
        for i, layer in enumerate(self.layers):
            layer.update(lr, weights_grads[i], bias_grads[i])

    def fit(self, x, y, epochs, lr):
        for epoch in range(epochs):
            running_loss = 0
            # print(f"input shape {x.shape}")
            for i, batch in enumerate(x):
                # print(f"batch shape {batch.shape}")
                y_pred = self.forward(batch)
                loss = self.loss(y[i], y_pred)

                # print(y_pred.shape)
                # print(y[i].shape)
                # print(loss.shape)
                loss_grad = self.loss(y, y_pred, grad=True)
                weights_grads, bias_grads = self.backward(loss_grad)
                self.update(lr, weights_grads, bias_grads)

                running_loss += loss

            print(f"Epoch {epoch}, Loss {running_loss / len(batch)}")

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    model = Model(mse)
    model.add(Linear(1, 4, ReLU()))
    model.add(Linear(4, 1, ReLU()))

    x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1, 1)
    y = np.sin(x).reshape(-1, 1, 1)

    # # random dummy data
    # x = np.random.rand(100, 1)
    # y = np.random.rand(100, 1)

    # # batch data into groups of 10
    # x = np.array_split(x, 10)
    # y = np.array_split(y, 10)

    print(f"Batch size: {len(x)}")
    print(f"Number of samples: {len(x[0])}")
    print(f"Data shape: {x[0].shape}")

    model.fit(x, y, 10, 0.01)
