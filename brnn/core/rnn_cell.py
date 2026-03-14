import numpy as np
from core.activation import tanh_derivative


class RNNCell:
    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.hidden_size = hidden_size

        # Xavier init for input weights: shape (hidden_size, input_size)
        # Connects each input feature to each hidden neuron
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        self.input_weights = np.random.randn(hidden_size, input_size) * scale

        # Xavier init for hidden weights: shape (hidden_size, hidden_size)
        # Connects previous hidden state to current, this is the "memory" weight
        scale = np.sqrt(1.0 / (hidden_size + hidden_size))
        self.hidden_weights = np.random.randn(hidden_size, hidden_size) * scale

        # Hidden bias initialized to zero, shape (hidden_size, 1)
        self.hidden_bias = np.zeros((hidden_size, 1))

    def forward(self, input_vec: np.ndarray, prev_hidden: np.ndarray) -> np.ndarray:
        # Formula: pre_activation = (input_weights @ input_vec) + (hidden_weights @ prev_hidden) + hidden_bias
        # Performs: (H, I) @ (I, 1) + (H, H) @ (H, 1) + (H, 1) = (H, 1)
        pre_activation = (
            self.input_weights @ input_vec
            + self.hidden_weights @ prev_hidden
            + self.hidden_bias
        )

        # Apply tanh to squash values between -1 and 1
        hidden_state = np.tanh(pre_activation)

        # Save inputs and pre-activation for use in backward()
        self.cache = {
            "input_vec": input_vec,
            "prev_hidden": prev_hidden,
            "pre_activation": pre_activation,
        }

        return hidden_state

    def backward(self, hidden_grad: np.ndarray, hidden_weights: np.ndarray) -> tuple:
        # Step 1: Gradient through tanh activation (chain rule)
        # Shape: (hidden_size, 1), same as pre_activation
        pre_act_grad = tanh_derivative(self.cache["pre_activation"]) * hidden_grad

        # Step 2: Gradient for input weights
        # (H, 1) @ (1, I) = (H, I), same shape as input_weights
        input_weight_grad = pre_act_grad @ self.cache["input_vec"].T

        # Step 3: Gradient for hidden weights (recurrent connection)
        # (H, 1) @ (1, H) = (H, H), same shape as hidden_weights
        hidden_weight_grad = pre_act_grad @ self.cache["prev_hidden"].T

        # Step 4: Gradient for hidden bias, same shape as hidden_bias
        hidden_bias_grad = pre_act_grad

        # Step 5: Gradient to pass back to the previous time step
        # (H, H).T @ (H, 1) = (H, 1)
        prev_hidden_grad = hidden_weights.T @ pre_act_grad

        return prev_hidden_grad, {
            "input_weight_grad": input_weight_grad,
            "hidden_weight_grad": hidden_weight_grad,
            "hidden_bias_grad": hidden_bias_grad,
        }
