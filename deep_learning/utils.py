import numpy as np


def one_hot_target(target_class: np.array, predicted_shape: tuple) -> np.array:
    target_class = target_class.flatten()

    num_classes = predicted_shape[-1]
    num_samples = target_class.shape[0]

    one_hot = np.zeros((num_samples, num_classes))
    one_hot[np.arange(num_samples), target_class] = 1

    return one_hot


def assert_shape(arr, expected_shape):
    assert (
        arr.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {arr.shape}"
