<h1 align="center">Encoder-Decoder RNN for Sequence Reversal</h1>

A from-scratch implementation of a Sequence-to-Sequence (Seq2Seq) Encoder-Decoder RNN in Python using only NumPy. The model learns to reverse fixed-length digit sequences, the "Mirror Mirror" protocol from the *Hanapin Mo Ang Bit* laboratory activity.

## Project Structure

```
encodec/
├── core/
│   ├── encoder.py                  # Encoder RNN cell (forward + BPTT)
│   └── decoder.py                  # Decoder RNN cell with softmax output (forward + BPTT)
├── data/
│   └── dataset.py                  # Sequence generation and one-hot encoding utilities
├── display/
│   ├── heatmap.py                  # Encoder hidden-state heatmap visualization
│   └── loss_plot.py                # Training loss curve plot
├── model/
│   └── seq2seq.py                  # Seq2Seq wrapper (forward, loss, backward, SGD, save/load)
├── output/
│   ├── model.npz                   # Saved trained weights
│   ├── loss_curve.png              # Training loss vs. epoch plot
│   └── hidden_state_heatmap.png    # Encoder hidden-state heatmap
├── main.py                         # Training and evaluation entry point
└── predict.py                      # Inference CLI (loads saved weights, no retraining)
```

## File Descriptions

- `encoder.py`: `Encoder` class. Reads the input sequence one token at a time, updating its hidden state via `h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)`. Returns the final hidden state as the context vector. Implements BPTT in `backward()`.
- `decoder.py`: `Decoder` class. Receives the context vector as its initial hidden state and generates the output sequence token by token using teacher forcing during training. Projects hidden states to vocabulary logits via softmax. Implements BPTT in `backward()`, returning the gradient w.r.t. the context vector to bridge back into the encoder.
- `dataset.py`: Defines `VOCABULARY_SIZE = 10` and `SEQUENCE_LENGTH = 5`. Provides `generate_sequences()` (random digit sequences paired with their reversals) and `one_hot_encode()` / `encode_sequence()` utilities.
- `seq2seq.py`: `Seq2Seq` class wiring encoder and decoder. Handles the full forward pass, cross-entropy loss computation, gradient clipping (max norm 5.0), SGD parameter updates, and weight serialization to `.npz`.
- `heatmap.py`: `plot_hidden_state_heatmap()`: builds a `(hidden_size × seq_len)` matrix from cached encoder hidden states and saves it as a `RdBu_r` heatmap PNG.
- `loss_plot.py`: `plot_loss_curve()`: plots average cross-entropy loss per epoch and saves it as a PNG.
- `main.py`: Entry point. Generates train/test datasets, initializes the model, runs training, saves visualizations, evaluates accuracy, and saves weights.
- `predict.py`: Inference CLI. Loads saved weights and runs the model on a user-supplied sequence without retraining. Supports `--sequence` flag and interactive mode.

## Architecture

- Input: one-hot vectors of size 10 (digits 0–9)
- Encoder hidden size: 64 neurons
- Decoder hidden size: 64 neurons
- Output: softmax over vocabulary size 10
- Optimizer: SGD, learning rate 0.01
- Gradient clipping: L2 norm clipped to 5.0
- Weight initialization: Xavier

## Dataset

- 1000 training samples / 200 test samples
- Sequences of length 5, digits drawn uniformly from 0–9
- Target is the exact reversal of the input (e.g., `[3, 2, 9, 4, 4]` -> `[4, 4, 9, 2, 3]`)
- All inputs converted to one-hot vectors before being fed to the encoder

## Training Configuration

| Hyperparameter   | Value |
|------------------|-------|
| Vocabulary size  | 10    |
| Sequence length  | 5     |
| Hidden size      | 64    |
| Learning rate    | 0.01  |
| Epochs           | 50    |
| Train samples    | 1000  |
| Test samples     | 200   |

## Usage

**Train the model:**
```bash
python encodec/main.py
```

**Run inference on a saved model:**
```bash
python encodec/predict.py --weights output/model.npz --sequence 4 1 9 2 7
```

Or run interactively:
```bash
python encodec/predict.py
```

## Results

The model was trained for 50 epochs on 1000 randomly generated digit sequences. Loss decreased steadily across training:

```
Epoch  10/50  |  Loss: 0.002344
Epoch  20/50  |  Loss: 0.000978
Epoch  30/50  |  Loss: 0.000613
Epoch  40/50  |  Loss: 0.000445
Epoch  50/50  |  Loss: 0.000348
```

On the 200-sample test set, the model achieved **200/200 (100.0%) accuracy**, correctly reversing every sequence.

Sample test outputs:
```
Input:     [3, 2, 9, 4, 4]
Expected:  [4, 4, 9, 2, 3]
Predicted: [4, 4, 9, 2, 3]

Input:     [9, 0, 5, 6, 8]
Expected:  [8, 6, 5, 0, 9]
Predicted: [8, 6, 5, 0, 9]

Input:     [7, 5, 2, 7, 6]
Expected:  [6, 7, 2, 5, 7]
Predicted: [6, 7, 2, 5, 7]
```

## Analysis Questions

**1. Did the model produce the correct output on test examples?**

Yes. The model correctly reversed all 200 test sequences, where it achieved 100% accuracy. This confirms that the encoder successfully compressed the full input sequence into a context vector that the decoder could use to reconstruct the sequence in reverse order.

**2. How fast did the loss decrease?**

Loss dropped sharply in the first 10 epochs (from an initial value down to ~0.002344 by epoch 10), then continued to decrease at a slower rate through the remaining epochs, reaching ~0.000348 by epoch 50. The steepest reduction happened early in training, which is consistent with the model quickly learning the general reversal pattern before fine-tuning on harder cases.

**3. What patterns appeared in the hidden-state heatmap?**

The heatmap shows the activation of each of the 64 encoder hidden units (rows) across the 5 input time steps (columns). Red indicates activations
near +1, blue near -1, and white near 0, all of which are outputs of the tanh activation function.

The hidden state shifts visibly at each time step, reflecting that the encoder updates its internal representation as each new digit is read. The
most pronounced change occurs at t=2 when the digit '9' is processed, where several units spike to near +1 or -1, suggesting they respond strongly to
that input. The final column (t=4) represents the context vector passed to the decoder, and it carries a mixed pattern of activations across all 64
units, encoding the full sequence in a compressed form.

**4. What does the result suggest about the strengths and limits of Encoder-Decoder RNNs?**

The perfect accuracy on this task shows that an Encoder-Decoder RNN can reliably learn fixed-length sequence transformations when the mapping is
deterministic and the sequences are short. The context vector is sufficient to encode a 5-token sequence for reversal.

That said, the architecture is arguably more than what this task requires. Reversing a short fixed-length sequence has no ambiguity and no
variable output length, so a simpler model could likely solve it just as well. The Encoder-Decoder design shows its real value in tasks where the
input and output lengths differ, or where the output depends on a non-trivial compression of the full input, like translation or summarization.

For longer sequences, the context vector becomes a bottleneck: compressing more information into a single fixed-size vector makes it harder for
the decoder to recover earlier tokens accurately. This is a known limitation of the basic Encoder-Decoder architecture, and it motivates extensions
like attention mechanisms, which allow the decoder to refer back to all encoder hidden states rather than relying solely on the final one.
