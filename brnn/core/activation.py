import numpy as np


def tanh_derivative(values: np.ndarray) -> np.ndarray:
    # Derivative of tanh: 1 - tanh(x)^2
    # Used in backward pass to undo the tanh activation
    return 1.0 - np.tanh(values) ** 2
