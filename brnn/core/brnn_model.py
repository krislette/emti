import numpy as np
from core.rnn_cell import RNNCell


class BidirectionalRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.hidden_size = hidden_size

        # Two RNN cells: one reads left to right, the other right to left
        self.forward_cell = RNNCell(input_size, hidden_size)
        self.backward_cell = RNNCell(input_size, hidden_size)

        # Output layer maps combined hidden state to a single predicted value
        # Xavier init, shape: (output_size, hidden_size * 2)
        scale = np.sqrt(1.0 / (hidden_size * 2 + output_size))
        self.output_weights = np.random.randn(output_size, hidden_size * 2) * scale

        # Output bias, shape: (output_size, 1)
        self.output_bias = np.zeros((output_size, 1))

    def feedforward(self, sequence: list) -> list:
        # T = number of time steps in the sequence
        T = len(sequence)
        forward_states = []
        backward_states = []
        prev_forward = np.zeros((self.hidden_size, 1))
        prev_backward = np.zeros((self.hidden_size, 1))
        self.forward_cache = []
        self.backward_cache = []

        # Run forward cell left to right through the sequence
        for t in range(T):
            prev_forward = self.forward_cell.forward(sequence[t], prev_forward)
            forward_states.append(prev_forward)
            self.forward_cache.append(self.forward_cell.cache.copy())

        # Run backward cell right to left through the sequence
        for t in reversed(range(T)):
            prev_backward = self.backward_cell.forward(sequence[t], prev_backward)
            backward_states.insert(0, prev_backward)
            self.backward_cache.insert(0, self.backward_cell.cache.copy())

        # Combine both directions at each time step
        # Shape per step: (hidden_size * 2, 1)
        self.combined_states = [
            np.vstack([forward_states[t], backward_states[t]]) for t in range(T)
        ]

        # Apply output layer at every time step
        outputs = [
            self.output_weights @ h + self.output_bias for h in self.combined_states
        ]

        return outputs

    def backpropagation(
        self,
        sequence: list,
        outputs: list,
        targets: list,
        lr: float = 0.001,
    ) -> None:
        T = len(sequence)
        mid = T // 2

        # Step 1: MSE gradient only at the middle position
        # All other positions get zero gradient, we only care about fixing the middle
        output_grads = [np.zeros_like(outputs[t]) for t in range(T)]
        output_grads[mid] = 2 * (outputs[mid] - targets[0])

        # Step 2: Output layer weight and bias gradients
        grad_output_weights = sum(
            g @ h.T for g, h in zip(output_grads, self.combined_states)
        )
        grad_output_bias = sum(output_grads)

        # Step 3: Gradient flowing back into the combined hidden states
        hidden_grads = [self.output_weights.T @ g for g in output_grads]

        # Step 4: Update output layer weights and bias
        self.output_weights -= lr * grad_output_weights
        self.output_bias -= lr * grad_output_bias

        # Step 5: BPTT through forward cell, right to left in time
        forward_grads = {
            "input_weight_grad": 0,
            "hidden_weight_grad": 0,
            "hidden_bias_grad": 0,
        }
        current_grad = np.zeros((self.hidden_size, 1))
        for t in reversed(range(T)):
            current_grad = hidden_grads[t][: self.hidden_size] + current_grad
            self.forward_cell.cache = self.forward_cache[t]
            current_grad, g = self.forward_cell.backward(
                current_grad, self.forward_cell.hidden_weights
            )
            for k in forward_grads:
                forward_grads[k] += g[k]

        # Step 6: BPTT through backward cell, left to right in time
        backward_grads = {
            "input_weight_grad": 0,
            "hidden_weight_grad": 0,
            "hidden_bias_grad": 0,
        }
        current_grad = np.zeros((self.hidden_size, 1))
        for t in range(T):
            current_grad = hidden_grads[t][self.hidden_size :] + current_grad
            self.backward_cell.cache = self.backward_cache[t]
            current_grad, g = self.backward_cell.backward(
                current_grad, self.backward_cell.hidden_weights
            )
            for k in backward_grads:
                backward_grads[k] += g[k]

        # Step 7: Update both RNN cell weights
        # new_weight = old_weight - learning_rate * gradient
        self.forward_cell.input_weights -= lr * forward_grads["input_weight_grad"]
        self.forward_cell.hidden_weights -= lr * forward_grads["hidden_weight_grad"]
        self.forward_cell.hidden_bias -= lr * forward_grads["hidden_bias_grad"]
        self.backward_cell.input_weights -= lr * backward_grads["input_weight_grad"]
        self.backward_cell.hidden_weights -= lr * backward_grads["hidden_weight_grad"]
        self.backward_cell.hidden_bias -= lr * backward_grads["hidden_bias_grad"]

    def loss(self, outputs: list, targets: list) -> float:
        # MSE at the middle position only
        mid = len(outputs) // 2
        return float((outputs[mid][0, 0] - targets[0][0, 0]) ** 2)
