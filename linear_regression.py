import numpy as np
import matplotlib.pyplot as plt
import loss_functions as ls
from layers import Layer

from sklearn.model_selection import train_test_split


class LinearRegression(Layer):
    def __init__(self, input_size, loss_function=ls.mse):
        super().__init__()
        self.input_size = input_size
        self.output_size = 1

        self.loss_function = loss_function

        self.weights = np.random.randn(input_size, self.output_size)
        self.bias = np.zeros((1, self.output_size))

    def __repr__(self):
        return f"Linear Layer (in {self.input_size}, out {self.output_size})"

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights)
        self.z = self.z + self.bias
        return self.z

    def backward(self, output_grad):
        """
        Performs backpropagation on the layer.

        Args:
            output_grad (np.ndarray): grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            pass
        """
        # Step 1: Compute gradients with respect to weights and biases.
        # The gradient for the weights is computed as the dot product of the transpose of the input

        output_grad = np.mean(output_grad, axis=1, keepdims=True)
        weights_grad = np.mean(np.dot(self.input.T, output_grad))

        # The gradient for the bias is the average of the gradients from the pre-activation over the batch.
        bias_grad = np.mean(np.sum(output_grad, axis=0, keepdims=True))

        return weights_grad, bias_grad

    def update(self, lr, weights_grad, bias_grad):
        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad

    def train(self, x_batched, y_batched, lr, epochs):
        for epoch in range(epochs):
            running_loss = 0
            for x, y in zip(x_batched, y_batched):
                y_pred = self.forward(x)
                loss = self.loss_function(y, y_pred)

                grad = self.loss_function(y, y_pred, grad=True)
                weights_grad, bias_grad = self.backward(grad)
                self.update(lr, weights_grad, bias_grad)

                running_loss += loss

            print(f"Epoch {epoch}: Loss {running_loss / len(x_batched)}")

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    # Generate some simple mock data
    np.random.seed(42)  # For reproducibility

    # Step 1: Define parameters for the line
    m = 2  # Slope
    b = 5  # Intercept

    # Step 2: Generate some input values (e.g., 500 data points)
    x = np.linspace(0, 10, 500)  # 500 values between 0 and 10

    # Step 3: Generate the corresponding output values using the line equation with noise
    noise = np.random.randn(500)  # Add random noise
    y = m * x + b + noise  # y = mx + b + noise

    # Step 4: Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Reshape the training data into batches
    batch_size = 10  # Define the batch size
    n_samples = x_train.shape[0]
    n_batches = n_samples // batch_size  # Number of batches

    # Reshape the training data into batches
    x_train_batches = x_train[: n_batches * batch_size].reshape(
        (n_batches, batch_size, 1)
    )
    y_train_batches = y_train[: n_batches * batch_size].reshape(
        (n_batches, batch_size, 1)
    )

    x_test_batches = x_test.reshape((x_test.shape[0], 1))
    y_test_batches = y_test.reshape((y_test.shape[0], 1))

    print("X Train Batches:", x_train_batches.shape)
    print("Y Train Batches:", y_train_batches.shape)

    # Step 5: Create a Linear Regression model
    model = LinearRegression(1)

    # Step 6: Train the model using the batched data (we'll train using the full x_train in practice)
    print(f"\nTraining the model with {n_batches} batches")
    print(f"Train input shape: {x_train_batches.shape}")
    print(f"Train output shape: {y_train_batches.shape}")
    model.train(x_train, y_train, lr=0.01, epochs=6)

    # Step 7: Make predictions on the test set
    print("\nMaking predictions on the test set")
    print(f"Test input shape: {x_test_batches.shape}")
    y_pred = model.predict(x_test_batches)

    # Step 8: Plot the results
    y_pred_train = model.predict(x_train.reshape(-1, 1))
    y_pred_test = model.predict(x_test.reshape(-1, 1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Training set plot
    ax[0].scatter(x_train, y_train, label="Actual")
    ax[0].plot(x_train, y_pred_train, color="red", label="Predicted")
    ax[0].set_title("Training Set")
    ax[0].legend()

    # Test set plot
    ax[1].scatter(x_test, y_test, label="Actual")
    ax[1].plot(x_test, y_pred_test, color="red", label="Predicted")
    ax[1].set_title("Test Set")
    ax[1].legend()

    plt.show()

