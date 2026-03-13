<h1 align="center">Bidirectional RNN for Sequence Denoising</h1>

A from-scratch implementation of a Bidirectional Recurrent Neural Network (BRNN) in Python using only NumPy. The model learns to denoise corrupted geometric sequences by predicting the clean middle value.

## Project Structure

```
brnn/
├── core/
│   ├── activation.py       # Activation functions (dtanh, sigmoid)
│   ├── rnn_cell.py         # Single RNN cell implementation
│   └── brnn_model.py       # Bidirectional RNN model
├── data/
│   └── data.py             # Data generation and preprocessing
└── main.py                 # Training, evaluation, and interactive demo
```

## File Descriptions

### `core/activation.py`
Contains activation functions and their derivatives:
- `tanh_derivative()` - Used for hidden state activations (derivative)

### `core/rnn_cell.py`
Implements a single RNN cell that:
- Maintains hidden state across time steps
- Performs forward pass: `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
- Computes gradients during backward pass
- Updates weights using gradient descent

### `core/brnn_model.py`
Implements the bidirectional RNN architecture:
- Contains two RNN cells: forward and backward
- Combines outputs from both directions
- Handles forward and backward propagation through time
- Computes loss (Mean Squared Error)

### `data/data.py`
Handles data generation:
- `make_clean_sequence()` - Generates geometric sequences
- `normalize()` - Scales sequences to [0, 1] range
- `add_noise()` - Corrupts the middle value with random noise
- `generate_data()` - Creates training/test datasets
- `make_demo_sequence()` - Creates sequences for visualization

### `main.py`
Main execution script:
- Trains the model on 1000 samples
- Evaluates on 200 test samples
- Provides interactive demo for custom inputs

## Architecture Details

### Network Structure

- Input Layer:    1 neuron  (single value at each time step)
- Hidden Layer:   16 neurons (bidirectional: 8 forward + 8 backward)
- Output Layer:   1 neuron  (predicted clean value)

### How Bidirectional Processing Works

#### Forward Pass (for sequence [2, 4, 6, 8, 10]):

**Step 1: Forward RNN processes left to right**

- Time 0: Input=2  -> Forward hidden state h_f[0]
- Time 1: Input=4  -> Forward hidden state h_f[1] (uses h_f[0])
- Time 2: Input=6  -> Forward hidden state h_f[2] (uses h_f[1])
- Time 3: Input=8  -> Forward hidden state h_f[3] (uses h_f[2])
- Time 4: Input=10 -> Forward hidden state h_f[4] (uses h_f[3])

**Step 2: Backward RNN processes right to left**

- Time 4: Input=10 -> Backward hidden state h_b[4]
- Time 3: Input=8  -> Backward hidden state h_b[3] (uses h_b[4])
- Time 2: Input=6  -> Backward hidden state h_b[2] (uses h_b[3])
- Time 1: Input=4  -> Backward hidden state h_b[1] (uses h_b[2])
- Time 0: Input=2  -> Backward hidden state h_b[0] (uses h_b[1])

**Step 3: Combine at each time step**

- Time 0: output[0] = combine(h_f[0], h_b[0])
- Time 1: output[1] = combine(h_f[1], h_b[1])
- Time 2: output[2] = combine(h_f[2], h_b[2])  <- Focus is on middle value
- Time 3: output[3] = combine(h_f[3], h_b[3])
- Time 4: output[4] = combine(h_f[4], h_b[4])

**Key Point**: The forward RNN doesn't jump from 2 to 10. It processes sequentially (2->4->6->8->10), while the backward RNN processes in reverse (10->8->6->4->2). At each position, both hidden states are combined.

#### Backward Pass (Backpropagation Through Time):

**Step 1: Calculate loss**

Loss = (predicted_middle - true_middle)²
> Only the middle position (time step 2) contributes to loss

**Step 2: Backpropagate through output layer**

Gradient flows from output[2] back to h_f[2] and h_b[2]

**Step 3: Backpropagate through time**

Forward RNN:  gradient flows 2 <- 1 <- 0
Backward RNN: gradient flows 2 -> 3 -> 4

**Step 4: Update weights**

All weights in both RNNs are updated based on accumulated gradients

### Training Process (Per Data Point)

1. **Generate corrupted sequence**: [2, 4, 6*, 8, 10] (6 is corrupted)
2. **Forward pass**: Process entire sequence bidirectionally
3. **Compute loss**: Compare predicted middle value to true value (6)
4. **Backward pass**: Compute gradients via backpropagation through time
5. **Update weights**: Apply gradient descent
6. **Move to next data point**: Repeat steps 1-5

## Dataset

**Training**: 1000 geometric sequences
- Start values: 1-10 (random integers)
- Ratios: 1.5-5.0 (random floats)
- Sequence length: 5 values
- Corruption: Middle value (index 2) has random noise added

**Test**: 200 geometric sequences (same distribution)

**Example**:

Clean:     [3, 6, 12, 24, 48]
Corrupted: [3, 6, 15.2*, 24, 48]  (* = corrupted)
Target:    Predict 12

## Hyperparameters

- **Epochs**: 10
- **Learning Rate**: 0.005
- **Hidden Size**: 16 (8 forward + 8 backward)
- **Noise Level**: 0.3 (30% of normalized value)
- **Loss Function**: Mean Squared Error (MSE)

## Limitations

1. **No extrapolation**: Model only works on sequences similar to training data
   - Trained on ratios 1.5-5.0, fails on ratio=10
   - Trained on start 1-10, struggles with start=50

2. **Single layer**: Simple architecture limits learning capacity

3. **Vanilla RNN**: Susceptible to vanishing gradients (though less problematic with short sequences)

4. **Fixed sequence length**: Only works with 5-element sequences

## Usage

```
bash
python main.py
```

The script will:
1. Train the model (10 epochs)
2. Show evaluation metrics
3. Enter interactive mode where you can test custom sequences
