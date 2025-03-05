import numpy as np
import matplotlib.pyplot as plt
from loss_functions import mse
from layers import Linear
from activation_functions import ReLU, Sigmoid, Tanh

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
            # print(f"Forward pass - Layer: {layer}, Output shape: {x.shape}")
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
            for i, batch in enumerate(x):
                y_pred = self.forward(batch)
                loss = self.loss(y[i], y_pred)

                loss_grad = self.loss(y[i], y_pred, grad=True)
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
    # Input layer
    model.add(Linear(1, 4, Tanh()))  # Increase hidden size to 32

    # Hidden layers
    model.add(Linear(4, 8, ReLU()))
    model.add(Linear(8, 16, ReLU()))
    model.add(Linear(16, 8, ReLU()))
    model.add(Linear(8, 4, ReLU()))

    # Output layer
    model.add(Linear(4, 1, Tanh()))

    print(model)

    # Generate sine wave data
    x = np.linspace(0, 1 * np.pi, 500).reshape(-1, 1)  # 10x more points
    y = np.sin(x) + np.random.normal(0, 0.01, x.shape)  # Add slight noise

    # Alternative: Use Min-Max Scaling (range [-1,1])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Visualize data distribution
    plt.scatter(x_train, y_train, label="Train", alpha=0.5)
    plt.scatter(x_test, y_test, label="Test", alpha=0.5)
    plt.legend()
    plt.show()

    print(f"Shape before batching: {x_train.shape}")

    # Set batch size
    batch_size = 24

    # Ensure the training data is evenly divisible by batch size
    num_batches = len(x_train) // batch_size
    x_train = x_train[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    y_train = y_train[: num_batches * batch_size].reshape(num_batches, batch_size, -1)

    print(f"Shape after batching: {x_train.shape}")

    # # random dummy data
    # x = np.random.rand(100, 1)
    # y = np.random.rand(100, 1)

    # # batch data into groups of 10
    # x = np.array_split(x, 10)
    # y = np.array_split(y, 10)

    print(f"Batch size: {len(x_train)}")
    print(f"Number of samples: {len(x[0])}")
    print(f"Data shape: {x[0].shape}")

    loss = model.fit(x_train, y_train, 100, 0.01)

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
