import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from loss_functions import mse
from layers import Linear
from activation_functions import ReLU, Sigmoid, Tanh

from visualise import visualize_predictions_over_epochs

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

    def forward(self, x, loggings=False):
        """
        Forward pass through the network

        Args:
            x (np.ndarray): Input data in the shape [batch_size, features]

        Returns:
            x (np.ndarray): Output data in the shape [batch_size, features]

        """
        if loggings:
            print(f"input shape {x.shape}")
        for layer in self.layers:
            x = layer.forward(x)
        if loggings:
            print(f"output shape {x.shape}")
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

    def fit(
        self,
        x,
        y,
        epochs,
        lr,
        logging_predictions=False,
        test_set_logging: np.ndarray = None,
    ):
        loss_graph = []
        all_train_predictions, all_test_predictions = [], []
        for epoch in range(epochs):
            running_loss = 0
            train_epoch_predictions, test_epoch_predictions = [], []
            for i, batch in enumerate(x):
                curr_y = y[i]
                y_hat = self.forward(batch)

                if y_hat.shape != y[i].shape:
                    curr_y = curr_y.reshape(y_hat.shape)

                loss = self.loss(curr_y, y_hat)

                loss_grad = self.loss(curr_y, y_hat, grad=True)
                weights_grads, bias_grads = self.backward(loss_grad)
                self.update(lr, weights_grads, bias_grads)

                train_epoch_predictions.append(y_hat)

                running_loss += loss

            if logging_predictions:
                for i, batch in enumerate(test_set_logging):
                    y_hat = self.forward(batch)
                    test_epoch_predictions.append(y_hat)

            if logging_predictions:
                all_train_predictions.append(train_epoch_predictions)
                all_test_predictions.append(test_epoch_predictions)

            print(f"Epoch {epoch}, Loss {running_loss / len(batch)}")
            loss_graph.append(running_loss / len(batch))

        if logging_predictions:
            return (
                loss_graph,
                np.array(all_train_predictions),
                np.array(all_test_predictions),
            )
        else:
            return loss_graph

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    # Create Model
    model = Model(mse)
    model.add(Linear(1, 4, Tanh()))
    model.add(Linear(4, 1, Tanh()))
    print(f"Model: {model}")

    # Generate sine wave data
    x = np.linspace(0, 1 * np.pi, 500).reshape(-1, 1)  # 10x more points
    y = np.sin(x) + np.random.normal(0, 0.02, x.shape)  # Add slight noise
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Get shape [samples, batch_size, features] but only stochastic gradient descent
    x_train = x_train.reshape(-1, 1, 1)
    y_train = y_train.reshape(-1, 1, 1)
    x_test = x_test.reshape(-1, 1, 1)
    y_test = y_test.reshape(-1, 1, 1)
    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    print(f"Test shape: {x_test.shape}, {y_test.shape}")

    # Train Model
    lr = 0.001
    loss, train_predictions, test_predictions = model.fit(
        x_train, y_train, 150, lr, logging_predictions=True, test_set_logging=x_test
    )
    plt.plot(loss)
    plt.show()

    visualize_predictions_over_epochs(
        f"visuals/train_sine_wave_lr{lr}.gif", train_predictions, x_train
    )
    visualize_predictions_over_epochs(
        f"visuals/test_sine_wave_{lr}.gif", test_predictions, x_test
    )

    # Test Model
    predictions = []
    for batch in x_test:
        predictions.append(model.predict(batch))

    predictions = np.array(predictions).reshape(-1, 1)
    plt.scatter(x_test, predictions, label="yhat")
    plt.scatter(x_test, y_test, label="y")
    plt.legend()
    plt.show()
