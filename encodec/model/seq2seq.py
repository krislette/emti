import numpy as np

from core.encoder import Encoder
from core.decoder import Decoder


class Seq2Seq:
    """
    Sequence-to-Sequence model for the Mirror Mirror protocol.

    Wires the Encoder and Decoder together into a single training unit:
      1. The Encoder reads the input sequence and produces a context vector.
      2. The Decoder takes the context vector and generates the output sequence.
      3. Cross-entropy loss is computed over all decoder output steps.
      4. Gradients are backpropagated through both the Decoder and Encoder.
      5. All parameters are updated with SGD.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        learning_rate: float,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Both encoder and decoder share the same input/output vocabulary
        self.encoder = Encoder(
            input_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.decoder = Decoder(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
        )

    def forward(
        self,
        input_sequence: list[np.ndarray],
        target_sequence: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list, np.ndarray, list]:
        """
        Full forward pass through both encoder and decoder.

        Args:
            input_sequence : one-hot encoded input,  list of (vocab_size, 1)
            target_sequence: one-hot encoded targets, list of (vocab_size, 1)

        Returns:
            probabilities_per_step : decoder softmax outputs at each step
            decoder_hidden_states  : decoder hidden states at each step (for heatmap)
            decoder_step_cache     : decoder cache for backprop
            context_vector         : encoder's final hidden state
            encoder_step_cache     : encoder cache for backprop
        """
        # Encoder
        # Reads input left to right, compresses into context vector
        context_vector, encoder_step_cache = self.encoder.forward(input_sequence)

        # Decoder
        # Starts from context vector, predicts reversed sequence token by token
        probabilities_per_step, decoder_hidden_states, decoder_step_cache = (
            self.decoder.forward(context_vector, target_sequence)
        )

        return (
            probabilities_per_step,
            decoder_hidden_states,
            decoder_step_cache,
            context_vector,
            encoder_step_cache,
        )

    def compute_loss(
        self,
        probabilities_per_step: list[np.ndarray],
        target_sequence: list[np.ndarray],
    ) -> float:
        """
        Compute average cross-entropy loss over all decoder output steps.

        Cross-entropy at step t:
          L_t = -sum( y_t * log(p_t) )

        Where y_t is the one-hot target and p_t is the predicted
        probability distribution.

        Args:
            probabilities_per_step: list of softmax outputs, each (vocab_size, 1)
            target_sequence       : list of one-hot targets,  each (vocab_size, 1)

        Returns:
            Average cross-entropy loss across all time steps (scalar float)
        """
        total_loss = 0.0

        for predicted_probs, target_vector in zip(
            probabilities_per_step, target_sequence
        ):
            # Clip probabilities to avoid log(0)
            clipped_probs = np.clip(predicted_probs, 1e-9, 1.0)
            total_loss += -np.sum(target_vector * np.log(clipped_probs))

        return total_loss / len(target_sequence)

    def _clip_gradient(self, gradient: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
        """
        Clip a gradient array by global norm to prevent exploding gradients.

        If the L2 norm of the gradient exceeds max_norm, the gradient is
        scaled down so its norm equals max_norm. Otherwise it is unchanged.

        This is especially important for RNNs, where gradients can grow
        exponentially as they propagate back through many time steps.

        Args:
            gradient : gradient array of any shape
            max_norm : maximum allowed L2 norm (default: 5.0)

        Returns:
            Clipped gradient array (same shape as input)
        """
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            return gradient * (max_norm / norm)
        return gradient

    def backward_and_update(
        self,
        encoder_step_cache: list,
        decoder_step_cache: list,
        target_sequence: list[np.ndarray],
    ) -> None:
        """
        Backpropagate gradients through decoder and encoder, then apply SGD.

        Flow:
          1. Decoder backward  -> gradients for decoder params + grad_context_vector
          2. Encoder backward  -> gradients for encoder params (using grad_context_vector)
          3. Gradient clipping -> prevent exploding gradients during BPTT
          4. SGD update        -> subtract lr * gradient from each parameter

        Args:
            encoder_step_cache: encoder cache from the forward pass
            decoder_step_cache: decoder cache from the forward pass
            target_sequence   : one-hot targets for loss gradient computation
        """
        # Step 1: Decoder backward pass
        # Returns decoder parameter gradients and the gradient w.r.t. the
        # context vector, which is the bridge back into the encoder
        decoder_gradients, grad_context_vector = self.decoder.backward(
            decoder_step_cache, target_sequence
        )

        # Step 2: Encoder backward pass
        # The gradient of the loss w.r.t. the context vector tells the encoder
        # how to adjust its weights to produce a better summary of the input
        encoder_gradients = self.encoder.backward(
            encoder_step_cache, grad_context_vector
        )

        # Step 3: SGD update for decoder parameters (with gradient clipping)
        self.decoder.input_to_hidden_weights -= (
            self.learning_rate
            * self._clip_gradient(decoder_gradients["grad_input_to_hidden_weights"])
        )
        self.decoder.hidden_to_hidden_weights -= (
            self.learning_rate
            * self._clip_gradient(decoder_gradients["grad_hidden_to_hidden_weights"])
        )
        self.decoder.hidden_bias -= self.learning_rate * self._clip_gradient(
            decoder_gradients["grad_hidden_bias"]
        )
        self.decoder.hidden_to_output_weights -= (
            self.learning_rate
            * self._clip_gradient(decoder_gradients["grad_hidden_to_output_weights"])
        )
        self.decoder.output_bias -= self.learning_rate * self._clip_gradient(
            decoder_gradients["grad_output_bias"]
        )

        # Step 4: SGD update for encoder parameters (with gradient clipping)
        self.encoder.input_to_hidden_weights -= (
            self.learning_rate
            * self._clip_gradient(encoder_gradients["grad_input_to_hidden_weights"])
        )
        self.encoder.hidden_to_hidden_weights -= (
            self.learning_rate
            * self._clip_gradient(encoder_gradients["grad_hidden_to_hidden_weights"])
        )
        self.encoder.hidden_bias -= self.learning_rate * self._clip_gradient(
            encoder_gradients["grad_hidden_bias"]
        )

    def predict(self, input_sequence: list[np.ndarray]) -> list[int]:
        """
        Run inference on a single input sequence (no teacher forcing).

        The decoder generates each token by feeding its own previous
        prediction as the next input.

        Args:
            input_sequence: one-hot encoded input, list of (vocab_size, 1)

        Returns:
            predicted_digits: list of predicted digit integers
        """
        # Encode the input into a context vector
        context_vector, _ = self.encoder.forward(input_sequence)

        hidden_state = context_vector
        predicted_digits = []
        decoder_input = np.zeros((self.vocab_size, 1))

        for _ in range(len(input_sequence)):
            prev_hidden_state = hidden_state

            pre_activation = (
                self.decoder.input_to_hidden_weights @ decoder_input
                + self.decoder.hidden_to_hidden_weights @ prev_hidden_state
                + self.decoder.hidden_bias
            )
            hidden_state = np.tanh(pre_activation)

            output_logits = (
                self.decoder.hidden_to_output_weights @ hidden_state
                + self.decoder.output_bias
            )

            # Pick the highest-probability token
            predicted_digit = int(np.argmax(output_logits))
            predicted_digits.append(predicted_digit)

            # Feed this prediction as the next decoder input (autoregressive)
            decoder_input = np.zeros((self.vocab_size, 1))
            decoder_input[predicted_digit] = 1.0

        return predicted_digits

    def save_weights(self, filepath: str) -> None:
        """
        Save all encoder and decoder weights to a .npz file.

        Args:
            filepath: path to save the weights (e.g., 'output/model.npz')
        """
        np.savez(
            filepath,
            # Encoder weights
            encoder_input_to_hidden_weights=self.encoder.input_to_hidden_weights,
            encoder_hidden_to_hidden_weights=self.encoder.hidden_to_hidden_weights,
            encoder_hidden_bias=self.encoder.hidden_bias,
            # Decoder weights
            decoder_input_to_hidden_weights=self.decoder.input_to_hidden_weights,
            decoder_hidden_to_hidden_weights=self.decoder.hidden_to_hidden_weights,
            decoder_hidden_bias=self.decoder.hidden_bias,
            decoder_hidden_to_output_weights=self.decoder.hidden_to_output_weights,
            decoder_output_bias=self.decoder.output_bias,
        )
        print(f"[model] Weights saved -> {filepath}.npz")

    def load_weights(self, filepath: str) -> None:
        """
        Load encoder and decoder weights from a saved .npz file.

        Args:
            filepath: path to the saved weights file (with or without .npz)
        """
        weights = np.load(filepath)

        # Restore encoder weights
        self.encoder.input_to_hidden_weights = weights[
            "encoder_input_to_hidden_weights"
        ]
        self.encoder.hidden_to_hidden_weights = weights[
            "encoder_hidden_to_hidden_weights"
        ]
        self.encoder.hidden_bias = weights["encoder_hidden_bias"]

        # Restore decoder weights
        self.decoder.input_to_hidden_weights = weights[
            "decoder_input_to_hidden_weights"
        ]
        self.decoder.hidden_to_hidden_weights = weights[
            "decoder_hidden_to_hidden_weights"
        ]
        self.decoder.hidden_bias = weights["decoder_hidden_bias"]
        self.decoder.hidden_to_output_weights = weights[
            "decoder_hidden_to_output_weights"
        ]
        self.decoder.output_bias = weights["decoder_output_bias"]

        print(f"[model] Weights loaded ← {filepath}")
