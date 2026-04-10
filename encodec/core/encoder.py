import numpy as np


class Encoder:
    """
    Encoder RNN.

    Reads the input sequence one token at a time, updating its hidden
    state at every step. After processing the full sequence, it passes
    its final hidden state as the context vector to the Decoder.

    Architecture:
      h_t = tanh(W_input_hidden * x_t + W_hidden_hidden * h_(t-1) + b_hidden)

    Where:
      x_t           : one-hot input vector at time t,  shape (input_size, 1)
      h_t           : hidden state at time t,           shape (hidden_size, 1)
      W_input_hidden: input-to-hidden weight matrix,    shape (hidden_size, input_size)
      W_hidden_hidden: hidden-to-hidden weight matrix,  shape (hidden_size, hidden_size)
      b_hidden      : hidden bias vector,               shape (hidden_size, 1)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier initialization for stable gradients
        input_scale = np.sqrt(1.0 / input_size)
        hidden_scale = np.sqrt(1.0 / hidden_size)

        # Weight matrix: input token -> hidden state
        # Shape: (hidden_size, input_size)
        self.input_to_hidden_weights = (
            np.random.randn(hidden_size, input_size) * input_scale
        )

        # Weight matrix: previous hidden state -> current hidden state
        # Shape: (hidden_size, hidden_size)
        self.hidden_to_hidden_weights = (
            np.random.randn(hidden_size, hidden_size) * hidden_scale
        )

        # Bias for the hidden state update
        # Shape: (hidden_size, 1)
        self.hidden_bias = np.zeros((hidden_size, 1))

    def forward(self, input_sequence: list[np.ndarray]) -> tuple[np.ndarray, list]:
        """
        Run the encoder over the full input sequence.

        At each time step t:
          pre_activation_t = W_input_hidden * x_t
                           + W_hidden_hidden * h_(t-1)
                           + b_hidden
          h_t = tanh(pre_activation_t)

        After all steps, the final hidden state h_T becomes the
        context vector that summarizes the entire input.

        Args:
            input_sequence: list of one-hot vectors, each shape (input_size, 1)

        Returns:
            context_vector : final hidden state h_T, shape (hidden_size, 1)
            step_cache     : list of per-step cache dicts for backprop
        """
        hidden_state = np.zeros((self.hidden_size, 1))
        step_cache = []

        for input_vector in input_sequence:
            prev_hidden_state = hidden_state

            # Compute pre-activation (linear combination before tanh)
            pre_activation = (
                self.input_to_hidden_weights @ input_vector
                + self.hidden_to_hidden_weights @ prev_hidden_state
                + self.hidden_bias
            )

            # Apply tanh activation to get new hidden state
            hidden_state = np.tanh(pre_activation)

            # Cache everything needed for the backward pass
            step_cache.append(
                {
                    "input_vector": input_vector,
                    "prev_hidden_state": prev_hidden_state,
                    "hidden_state": hidden_state,
                }
            )

        context_vector = hidden_state
        return context_vector, step_cache

    def backward(
        self,
        step_cache: list,
        incoming_hidden_gradient: np.ndarray,
    ) -> dict:
        """
        Backpropagate gradients through the encoder (BPTT).

        Starting from the gradient of the loss w.r.t. the context
        vector, propagate backwards through every encoder time step.

        Args:
            step_cache              : cached values from the forward pass
            incoming_hidden_gradient: dL/d(context_vector), shape (hidden_size, 1)

        Returns:
            gradients: dict containing accumulated gradients for all
                       encoder parameters
        """
        grad_input_to_hidden_weights = np.zeros_like(self.input_to_hidden_weights)
        grad_hidden_to_hidden_weights = np.zeros_like(self.hidden_to_hidden_weights)
        grad_hidden_bias = np.zeros_like(self.hidden_bias)

        # Gradient flowing back in time starts as the incoming gradient
        grad_hidden_state = incoming_hidden_gradient

        for cache in reversed(step_cache):
            input_vector = cache["input_vector"]
            prev_hidden_state = cache["prev_hidden_state"]
            hidden_state = cache["hidden_state"]

            # Backprop through tanh: d/dx tanh(x) = 1 - tanh(x)^2
            grad_pre_activation = (1.0 - hidden_state**2) * grad_hidden_state

            # Accumulate gradients for each parameter
            grad_input_to_hidden_weights += grad_pre_activation @ input_vector.T
            grad_hidden_to_hidden_weights += grad_pre_activation @ prev_hidden_state.T
            grad_hidden_bias += grad_pre_activation

            # Pass gradient further back in time to the previous hidden state
            grad_hidden_state = self.hidden_to_hidden_weights.T @ grad_pre_activation

        return {
            "grad_input_to_hidden_weights": grad_input_to_hidden_weights,
            "grad_hidden_to_hidden_weights": grad_hidden_to_hidden_weights,
            "grad_hidden_bias": grad_hidden_bias,
        }
