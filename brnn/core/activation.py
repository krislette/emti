import numpy as np


def tanh_derivative(values: np.ndarray) -> np.ndarray:
    # Derivative of tanh: 1 - tanh(x)^2
    # Used in backward pass to undo the tanh activation
    return 1.0 - np.tanh(values) ** 2


def softmax(values: np.ndarray) -> np.ndarray:
    # Subtract max first to avoid overflow in exp()
    shifted = np.exp(values - np.max(values))

    # Divide by sum so all values add up to 1.0 (probabilities)
    return shifted / shifted.sum()
