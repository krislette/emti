import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over a column vector.

    Subtracts the max before exponentiating to prevent overflow.

    Input shape : (vocab_size, 1)
    Output shape: (vocab_size, 1), sums to 1.0
    """
    shifted = logits - np.max(logits)
    exponents = np.exp(shifted)
    return exponents / np.sum(exponents)


class Decoder:
    """
    Decoder RNN.

    Receives the encoder's context vector as its initial hidden state,
    then generates the output sequence one token at a time.

    At each step the decoder:
      1. Updates its hidden state using the previous hidden state and
         the previous output token (teacher forcing during training).
      2. Projects the hidden state to logits over the vocabulary.
      3. Applies softmax to get a probability distribution.

    Architecture:
      h_t   = tanh(W_input_hidden * x_t + W_hidden_hidden * h_(t-1) + b_hidden)
      y_t   = W_hidden_output * h_t + b_output
      p_t   = softmax(y_t)

    Where:
      x_t              : input at time t (previous target, one-hot),  shape (input_size, 1)
      h_t              : hidden state at time t,                       shape (hidden_size, 1)
      y_t              : output logits at time t,                      shape (vocab_size, 1)
      p_t              : probability distribution at time t,           shape (vocab_size, 1)
      W_input_hidden   : input-to-hidden weight matrix,                shape (hidden_size, input_size)
      W_hidden_hidden  : hidden-to-hidden weight matrix,               shape (hidden_size, hidden_size)
      b_hidden         : hidden bias,                                   shape (hidden_size, 1)
      W_hidden_output  : hidden-to-output weight matrix,               shape (vocab_size, hidden_size)
      b_output         : output bias,                                   shape (vocab_size, 1)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier initialization
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

        # Weight matrix: hidden state -> output logits
        # Shape: (output_size, hidden_size)
        self.hidden_to_output_weights = (
            np.random.randn(output_size, hidden_size) * hidden_scale
        )

        # Bias for the output projection
        # Shape: (output_size, 1)
        self.output_bias = np.zeros((output_size, 1))

    def forward(
        self,
        context_vector: np.ndarray,
        target_sequence: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list]:
        """
        Run the decoder over the target sequence (teacher forcing).

        The context vector from the encoder seeds the decoder's hidden
        state.  At each step, the previous target token is fed as input
        (teacher forcing), so the decoder learns to predict the next
        token in the reversed sequence.

        Step 0 uses a zero vector as the input (no previous token yet).

        Args:
            context_vector : encoder's final hidden state, shape (hidden_size, 1)
            target_sequence: list of one-hot target vectors, each (output_size, 1)

        Returns:
            probabilities_per_step: list of softmax probability vectors
            hidden_states_per_step: list of hidden state vectors (for heatmap)
            step_cache            : list of per-step cache dicts for backprop
        """
        hidden_state = context_vector
        probabilities_per_step = []
        hidden_states_per_step = []
        step_cache = []

        for step_index, target_vector in enumerate(target_sequence):
            # Teacher forcing: feed the previous target token as input.
            # At step 0, there is no previous token so we use a zero vector.
            if step_index == 0:
                decoder_input = np.zeros((self.input_size, 1))
            else:
                decoder_input = target_sequence[step_index - 1]

            prev_hidden_state = hidden_state

            # Compute pre-activation and update hidden state
            pre_activation = (
                self.input_to_hidden_weights @ decoder_input
                + self.hidden_to_hidden_weights @ prev_hidden_state
                + self.hidden_bias
            )
            hidden_state = np.tanh(pre_activation)

            # Project hidden state to output logits
            output_logits = (
                self.hidden_to_output_weights @ hidden_state + self.output_bias
            )

            # Convert logits to probability distribution
            output_probabilities = softmax(output_logits)

            probabilities_per_step.append(output_probabilities)
            hidden_states_per_step.append(hidden_state)

            # Cache values needed for the backward pass at this step
            step_cache.append(
                {
                    "decoder_input": decoder_input,
                    "prev_hidden_state": prev_hidden_state,
                    "hidden_state": hidden_state,
                    "output_logits": output_logits,
                    "output_probabilities": output_probabilities,
                }
            )

        return probabilities_per_step, hidden_states_per_step, step_cache

    def backward(
        self,
        step_cache: list,
        target_sequence: list[np.ndarray],
    ) -> tuple[dict, np.ndarray]:
        """
        Backpropagate gradients through the decoder (BPTT).

        Uses cross-entropy loss gradient (softmax + cross-entropy
        combined derivative simplifies to: p_t - y_t).

        Args:
            step_cache      : cached values from the forward pass
            target_sequence : list of one-hot target vectors

        Returns:
            gradients              : dict of accumulated parameter gradients
            grad_context_vector    : gradient w.r.t. the context vector
                                     (passed back to the encoder)
        """
        grad_input_to_hidden_weights = np.zeros_like(self.input_to_hidden_weights)
        grad_hidden_to_hidden_weights = np.zeros_like(self.hidden_to_hidden_weights)
        grad_hidden_bias = np.zeros_like(self.hidden_bias)
        grad_hidden_to_output_weights = np.zeros_like(self.hidden_to_output_weights)
        grad_output_bias = np.zeros_like(self.output_bias)

        # No gradient flows back in time past the last step initially
        grad_hidden_state = np.zeros((self.hidden_size, 1))

        for step_index in reversed(range(len(step_cache))):
            cache = step_cache[step_index]
            target_vector = target_sequence[step_index]

            decoder_input = cache["decoder_input"]
            prev_hidden_state = cache["prev_hidden_state"]
            hidden_state = cache["hidden_state"]
            output_probabilities = cache["output_probabilities"]

            # [] Output layer gradients
            # Cross-entropy + softmax combined gradient: dL/d(logits) = p_t - y_t
            grad_output_logits = output_probabilities - target_vector

            grad_hidden_to_output_weights += grad_output_logits @ hidden_state.T
            grad_output_bias += grad_output_logits

            # Gradient flowing into the hidden state from the output layer
            grad_hidden_from_output = (
                self.hidden_to_output_weights.T @ grad_output_logits
            )

            # Combine with gradient flowing back in time from the next step
            grad_hidden_total = grad_hidden_from_output + grad_hidden_state

            # [] Hidden state gradients (backprop through tanh)
            grad_pre_activation = (1.0 - hidden_state**2) * grad_hidden_total

            grad_input_to_hidden_weights += grad_pre_activation @ decoder_input.T
            grad_hidden_to_hidden_weights += grad_pre_activation @ prev_hidden_state.T
            grad_hidden_bias += grad_pre_activation

            # Pass gradient to the previous hidden state (back in time)
            grad_hidden_state = self.hidden_to_hidden_weights.T @ grad_pre_activation

        # The gradient that has propagated all the way back to h_0 (context vector)
        grad_context_vector = grad_hidden_state

        gradients = {
            "grad_input_to_hidden_weights": grad_input_to_hidden_weights,
            "grad_hidden_to_hidden_weights": grad_hidden_to_hidden_weights,
            "grad_hidden_bias": grad_hidden_bias,
            "grad_hidden_to_output_weights": grad_hidden_to_output_weights,
            "grad_output_bias": grad_output_bias,
        }

        return gradients, grad_context_vector
