import numpy as np
import matplotlib.pyplot as plt
from loss_functions import mse
from layers import Linear
from activation_functions import ReLU, Sigmoid, Tanh


class Model:
    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            print(f"Forward pass - Layer: {layer}, Output shape: {x.shape}")
        return x

    def backward(self, loss_grad):
        weights_grads = []
        bias_grads = []
        prior_loss = loss_grad
        for layer in reversed(self.layers):
            next_grad, weights_grad, bias_grad = layer.backward(prior_loss)

            # print(f"Backward pass - Layer: {layer}, Gradient shape: {next_grad.shape}")

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
        loss_graph = []
        for epoch in range(epochs):
            running_loss = 0
            # print(f"input shape {x.shape}")
            for i, batch in enumerate(x):
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
            loss_graph.append(running_loss / len(batch))

        return loss_graph

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    model = Model(mse)
    model.add(
        Linear(1, 16, Sigmoid())
    )  # Input size is 1, first hidden layer size is 16
    model.add(
        Linear(16, 32, Sigmoid())
    )  # First hidden layer size is 16, second hidden layer size is 32
    model.add(
        Linear(32, 64, Sigmoid())
    )  # Second hidden layer size is 32, third hidden layer size is 64
    model.add(
        Linear(64, 32, Sigmoid())
    )  # Third hidden layer size is 64, fourth hidden layer size is 32
    model.add(
        Linear(32, 16, Sigmoid())
    )  # Fourth hidden layer size is 32, fifth hidden layer size is 16
    model.add(
        Linear(16, 8, Sigmoid())
    )  # Fifth hidden layer size is 16, sixth hidden layer size is 8
    model.add(
        Linear(8, 1, Sigmoid())
    )  # Sixth hidden layer size is 8, output layer size is 1

    print(model)

    # Generate sine wave data
    x = np.linspace(0, 2 * np.pi, 2050)
    x = x.reshape(-1, 1)
    y = np.sin(x)

    # split into train and test
    x_test, x = np.split(x, [50])
    y_test, y = np.split(y, [50])

    print(f"Shape before batching: {x.shape}")

    # shuffle x and y
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # batch data into groups of 10
    x = np.array(np.array_split(x, len(x) // 200))
    y = np.array(np.array_split(y, len(y) // 200))

    print(f"Shape after batching: {x.shape}")

    # # random dummy data
    # x = np.random.rand(100, 1)
    # y = np.random.rand(100, 1)

    # # batch data into groups of 10
    # x = np.array_split(x, 10)
    # y = np.array_split(y, 10)

    print(f"Batch size: {len(x)}")
    print(f"Number of samples: {len(x[0])}")
    print(f"Data shape: {x[0].shape}")

    loss = model.fit(x, y, 100, 0.1)

    plt.plot(loss)
    plt.show()

    print(f"Test Size: {len(x_test)}")
    predictions = []
    for batch in x_test:
        predictions.append(model.predict(batch))

    predictions = np.array(predictions).reshape(-1, 1)
    plt.scatter(x_test, predictions, label="Predictions")
    plt.scatter(x_test, y_test, label="Ground truth")
    plt.legend()
    plt.show()
