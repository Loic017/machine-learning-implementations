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

from torch.utils.data import DataLoader

from utils import one_hot_target, assert_shape


class Model:
    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def __repr__(self):
        return "\n".join([str(layer) for layer in self.layers])

    def add(self, layer) -> None:
        """
        Add a layer to the model.

        Args:
            layer (Layer): Layer to be added to the model.

        Returns:
            None
        """
        self.layers.append(layer)

    def forward(self, x, loggings=False) -> np.ndarray:
        """
        Forward pass through the network

        Args:
            x (np.ndarray): Input data in the shape [batch_size, features]

        Returns:
            yhat (np.ndarray): Output data in the shape [batch_size, features]
        """
        original_input = x

        for i, layer in enumerate(self.layers):
            assert_shape(arr=x, expected_shape=(x.shape[0], layer.input_size))
            x = layer.forward(x)

        if loggings:
            print(f"input shape {x.shape}")
            print(f"output shape {original_input.shape}")
        return x

    def backward(self, loss_grad) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute backpropagation on the model.

        Args:
            loss_grad (np.ndarray): Gradient of the loss with respect to the model's output.

        Returns:
            weights_grad (list): Gradient of the loss with respect to the weights of each layer.
            bias_grad (list): Gradient of the loss with respect to the bias of each layer.
        """
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
        train_data: tuple[np.ndarray, np.ndarray],
        epochs,
        lr,
        batch_size,
        validation_data: tuple[np.ndarray, np.ndarray] = None,
        logging_predictions=False,
        test_set_logging: np.ndarray = None,
    ):
        """
        Train the model.

        Args:
            train_data (tuple[np.ndarray, np.ndarray]): Tuple of input data and target data.
            epochs (int): Number of epochs to train for.
            lr (float): Learning rate.
            validation_data (tuple[np.ndarray, np.ndarray], optional): Tuple of validation input data and target data. Defaults to None.
            logging_predictions (bool, optional): Whether to log predictions. Defaults to False.
            test_set_logging (np.ndarray, optional): Test set for logging predictions. Defaults to None.
        """
        x = train_data[0]
        y = train_data[1]

        assert_shape(
            arr=x,
            expected_shape=(x.shape[0], self.layers[0].input_size),
        )

        if validation_data is not None:
            x_val = validation_data[0]
            y_val = validation_data[1]
            val_loader = DataLoader(
                dataset=list(zip(x_val, y_val)),
                batch_size=128,
                shuffle=False,
                drop_last=True,
            )

        loss_graph = {
            "train": [],
            "val": [],
        }
        all_train_predictions, all_test_predictions = [], []

        dataloader = DataLoader(
            dataset=list(zip(x, y)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        assert_shape(
            arr=np.array(next(iter(dataloader))[0]),
            expected_shape=(batch_size, self.layers[0].input_size),
        )

        for epoch in range(epochs):
            running_loss = 0
            train_epoch_predictions, test_epoch_predictions = [], []
            print(f"Training epoch {epoch}")
            for batch in dataloader:
                x, y = batch

                x = x.detach().numpy()
                y = y.detach().numpy()

                assert_shape(
                    arr=x,
                    expected_shape=(batch_size, self.layers[0].input_size),
                )

                # FORWARD PASS -> Expects shape [batch_size, features]
                y_hat = self.forward(x)

                # If y is not the same shape as y_hat, convert it to one-hot encoding

                # if y_hat.shape != y.shape:
                #     # y = one_hot_target(y, y_hat.shape)
                #     y_hat.squeeze()

                # Compute loss
                loss = self.loss(y, y_hat)  # Loss
                loss_grad = self.loss(
                    y, y_hat, grad=True
                )  # Gradient of the loss w.r.t y_hat

                # Backpropagation and update parameters
                weights_grads, bias_grads = self.backward(loss_grad)
                self.update(lr, weights_grads, bias_grads)

                # Rest is logging
                train_epoch_predictions.append(y_hat)

                running_loss += loss

            if logging_predictions:
                testlog_loader = DataLoader(
                    dataset=list(zip(x, np.zeros(len(x)))),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                for batch in testlog_loader:
                    x, _ = batch
                    y_hat = self.forward(x)
                    test_epoch_predictions.append(y_hat)

            if logging_predictions:
                all_train_predictions.append(train_epoch_predictions)
                all_test_predictions.append(test_epoch_predictions)

            print(f"Epoch {epoch} loss -> {running_loss / len(batch)}")
            loss_graph["train"].append(running_loss / len(batch))

            if validation_data is not None:
                running_val_loss = 0
                for batch in val_loader:
                    x_val, y_val = batch
                    y_hat = self.forward(x_val)
                    if y_hat.shape != y_val.shape:
                        y_val = one_hot_target(y_val, y_hat.shape)
                    loss = self.loss(y_val, y_hat)
                    running_val_loss += loss

                print(f"Validation loss -> {running_val_loss / len(batch)}")
                loss_graph["val"].append(running_val_loss / len(batch))

        if logging_predictions:
            return (
                loss_graph,
                np.array(all_train_predictions),
                np.array(all_test_predictions),
            )
        else:
            return loss_graph

    def predict(self, x):
        """
        Conduct an instance of forward pass on the model with input data x.

        Args:
            x (np.ndarray): Input data in the shape [batch_size, features]

        Returns:
            yhat (np.ndarray): Output data in the shape [batch_size, features]
        """
        # x = np.expand_dims(x, axis=0)
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
    # x_train = x_train.reshape(-1, 1, 1)
    # y_train = y_train.reshape(-1, 1, 1)
    # x_test = x_test.reshape(-1, 1, 1)
    # y_test = y_test.reshape(-1, 1, 1)
    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    print(f"Test shape: {x_test.shape}, {y_test.shape}")

    # Train Model
    lr = 0.5
    loss, train_predictions, test_predictions = model.fit(
        (x_train, y_train),
        100,
        lr,
        batch_size=32,
        logging_predictions=True,
        test_set_logging=x_test,
    )
    plt.plot(loss["train"], label="Training Loss")
    if loss["val"]:
        plt.plot(loss["val"], label="Validation Loss")
        plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.show()

    train_predictions_reshaped = train_predictions.reshape(
        train_predictions.shape[0], -1
    )
    test_predictions_reshaped = test_predictions.reshape(test_predictions.shape[0], -1)

    # visualize_predictions_over_epochs(
    #     f"visuals/train_sine_wave_lr{lr}.gif", train_predictions, x_train
    # )
    # visualize_predictions_over_epochs(
    #     f"visuals/test_sine_wave_{lr}.gif", test_predictions, x_test
    # )

    # Test Model
    predictions = []
    for batch in x_test:
        predictions.append(model.predict(np.expand_dims(batch, axis=0)))

    predictions = np.array(predictions).reshape(-1, 1)
    plt.scatter(x_test, predictions, label="yhat")
    plt.scatter(x_test, y_test, label="y")
    plt.legend()
    plt.show()
