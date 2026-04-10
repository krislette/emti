import numpy as np
import os

from data.dataset import (
    generate_sequences,
    encode_sequence,
    VOCABULARY_SIZE,
    SEQUENCE_LENGTH,
)
from model.seq2seq import Seq2Seq
from display.loss_plot import plot_loss_curve
from display.heatmap import plot_hidden_state_heatmap


HIDDEN_SIZE = 64  # Number of hidden units in both encoder and decoder
LEARNING_RATE = 0.01  # Fixed SGD learning rate (as required by the lab)
NUM_EPOCHS = 50  # Training iterations over the full dataset
NUM_TRAIN = 1000  # Number of training samples
NUM_TEST = 200  # Number of test samples to evaluate after training
LOG_INTERVAL = 10  # Print average loss every N epochs
OUTPUT_DIR = "output"


def train(model: Seq2Seq, dataset: list) -> list[float]:
    """
    Train the Seq2Seq model over all epochs.

    Each epoch iterates over every sample in the dataset:
      1. Encode the input sequence.
      2. Decode into the predicted reversed sequence.
      3. Compute cross-entropy loss.
      4. Backpropagate and update all parameters with SGD.

    Args:
        model  : the Seq2Seq model instance
        dataset: list of (input_digits, target_digits) tuples

    Returns:
        loss_history: average loss recorded after each epoch
    """
    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0

        for input_digits, target_digits in dataset:
            # Convert digit sequences to one-hot encoded vectors
            input_sequence = encode_sequence(input_digits)
            target_sequence = encode_sequence(target_digits)

            # Forward pass
            (
                probabilities_per_step,
                _decoder_hidden_states,
                decoder_step_cache,
                _context_vector,
                encoder_step_cache,
            ) = model.forward(input_sequence, target_sequence)

            # Loss computation
            loss = model.compute_loss(probabilities_per_step, target_sequence)
            epoch_loss += loss

            # Backward pass + SGD update
            model.backward_and_update(
                encoder_step_cache,
                decoder_step_cache,
                target_sequence,
            )

        average_epoch_loss = epoch_loss / len(dataset)
        loss_history.append(average_epoch_loss)

        if epoch % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch:>4}/{NUM_EPOCHS}  |  Loss: {average_epoch_loss:.6f}")

    return loss_history


def evaluate(model: Seq2Seq, test_dataset: list) -> None:
    """
    Run the trained model on unseen test sequences and print results.

    Args:
        model       : the trained Seq2Seq model
        test_dataset: list of (input_digits, target_digits) tuples
    """
    print("\n" + "=" * 55)
    print("  Test Results")
    print("=" * 55)

    num_correct = 0

    for input_digits, target_digits in test_dataset:
        input_sequence = encode_sequence(input_digits)
        predicted_digits = model.predict(input_sequence)

        is_correct = predicted_digits == target_digits
        status = "✓" if is_correct else "✗"

        if is_correct:
            num_correct += 1

        print(
            f"  {status}  Input:     {input_digits}\n"
            f"      Expected:  {target_digits}\n"
            f"      Predicted: {predicted_digits}\n"
        )

    accuracy = num_correct / len(test_dataset) * 100
    print(f"  Accuracy: {num_correct}/{len(test_dataset)} ({accuracy:.1f}%)")
    print("=" * 55)


def visualize(model: Seq2Seq, sample_input_digits: list[int]) -> None:
    """
    Generate and save the loss curve and encoder hidden-state heatmap.

    The heatmap is generated from a single forward pass through the
    encoder using the provided sample input.

    Args:
        model              : the trained Seq2Seq model
        sample_input_digits: one input sequence to visualize the encoder for
    """
    sample_input_sequence = encode_sequence(sample_input_digits)
    _context_vector, encoder_step_cache = model.encoder.forward(sample_input_sequence)

    plot_hidden_state_heatmap(
        encoder_step_cache=encoder_step_cache,
        input_digits=sample_input_digits,
        output_dir=OUTPUT_DIR,
    )


def main() -> None:
    print("=" * 55)
    print("  Mission: Hanapin Mo Ang Bit")
    print("  Protocol: Mirror Mirror")
    print("=" * 55)
    print(f"\n  Vocab size      : {VOCABULARY_SIZE}")
    print(f"  Sequence length : {SEQUENCE_LENGTH}")
    print(f"  Hidden size     : {HIDDEN_SIZE}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Train samples   : {NUM_TRAIN}")
    print(f"  Test samples    : {NUM_TEST}\n")

    # Data prep
    print("[data] Generating sequences...")
    train_dataset = generate_sequences(num_samples=NUM_TRAIN)
    test_dataset = generate_sequences(num_samples=NUM_TEST)

    # Model init
    print("[model] Initializing Seq2Seq (Encoder-Decoder RNN)...")
    model = Seq2Seq(
        vocab_size=VOCABULARY_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
    )

    # Training
    print(f"\n[train] Starting training for {NUM_EPOCHS} epochs...\n")
    loss_history = train(model, train_dataset)

    # Save loss curve
    print("\n[display] Generating visualizations...")
    plot_loss_curve(loss_history=loss_history, output_dir=OUTPUT_DIR)

    # Save hidden state heatmap (using first test sample as illustration)
    sample_input_digits = test_dataset[0][0]
    visualize(model, sample_input_digits)

    # Evaluation
    evaluate(model, test_dataset)

    print("\n[display] Generating confusion matrix...")

    # Save trained weights
    model.save_weights(os.path.join(OUTPUT_DIR, "model"))


if __name__ == "__main__":
    np.random.seed(42)
    main()
