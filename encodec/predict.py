"""
Loads saved model weights and runs inference on custom input sequences
without needing to retrain.

Usage:
    python predict.py
    python predict.py --weights output/model.npz --sequence 4 1 9 2 7
"""

import argparse
import numpy as np

from data.dataset import (
    encode_sequence,
    VOCABULARY_SIZE,
    SEQUENCE_LENGTH,
)
from model.seq2seq import Seq2Seq


DEFAULT_WEIGHTS_PATH = "output/model.npz"
HIDDEN_SIZE = 64  # Must match the value used during training


def load_model(weights_path: str) -> Seq2Seq:
    """
    Initialize a Seq2Seq model with the same architecture used during
    training and load the saved weights into it.

    Args:
        weights_path: path to the saved .npz weights file

    Returns:
        model: Seq2Seq instance with restored weights, ready for inference
    """
    model = Seq2Seq(
        vocab_size=VOCABULARY_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=0.0,  # Not needed for inference, set to 0
    )
    model.load_weights(weights_path)
    return model


def predict_single(model: Seq2Seq, input_digits: list[int]) -> None:
    """
    Run inference on a single input sequence and print the result.

    Args:
        model       : loaded Seq2Seq model
        input_digits: list of digit integers, e.g. [4, 1, 9, 2, 7]
    """
    if len(input_digits) != SEQUENCE_LENGTH:
        print(
            f"[error] Input must be exactly {SEQUENCE_LENGTH} digits. "
            f"Got {len(input_digits)}: {input_digits}"
        )
        return

    if any(d < 0 or d > 9 for d in input_digits):
        print(f"[error] All digits must be in range 0–9. Got: {input_digits}")
        return

    input_sequence = encode_sequence(input_digits)
    predicted_digits = model.predict(input_sequence)
    expected_digits = list(reversed(input_digits))
    is_correct = predicted_digits == expected_digits

    status = "✓ Correct" if is_correct else "✗ Incorrect"

    print(f"\n  Input    : {input_digits}")
    print(f"  Expected : {expected_digits}")
    print(f"  Predicted: {predicted_digits}")
    print(f"  Result   : {status}\n")


def interactive_mode(model: Seq2Seq) -> None:
    """
    Run an interactive loop where the user can type sequences and see
    predictions in real time. Type 'exit' or 'quit' to stop.

    Args:
        model: loaded Seq2Seq model
    """
    print("\n" + "=" * 55)
    print("  Mirror Mirror: Interactive Inference")
    print(f"  Enter {SEQUENCE_LENGTH} digits (0-9) separated by spaces.")
    print("  Type 'exit' to quit.")
    print("=" * 55)

    while True:
        try:
            raw_input = input("\n  Input sequence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[predict] Exiting.")
            break

        if raw_input.lower() in ("exit", "quit", "q"):
            print("[predict] Exiting.")
            break

        # Parse input
        try:
            input_digits = [int(token) for token in raw_input.split()]
        except ValueError:
            print(f"  [error] Could not parse '{raw_input}'. Please enter digits only.")
            continue

        predict_single(model, input_digits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror Mirror: Run inference on a trained Seq2Seq model."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to the saved .npz weights file (default: {DEFAULT_WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=None,
        help=(
            f"A sequence of {SEQUENCE_LENGTH} digits (0–9) to reverse. "
            "If omitted, launches interactive mode."
        ),
    )
    args = parser.parse_args()

    # Load the saved model
    print(f"\n[predict] Loading weights from '{args.weights}'...")
    model = load_model(args.weights)
    print(
        f"[predict] Model ready. Vocab size: {VOCABULARY_SIZE}, "
        f"\n  Hidden size: {HIDDEN_SIZE}\n  Sequence length: {SEQUENCE_LENGTH}"
    )

    if args.sequence is not None:
        # Single prediction from command-line argument
        predict_single(model, args.sequence)
    else:
        # No sequence provided, launch interactive mode
        interactive_mode(model)


if __name__ == "__main__":
    np.random.seed(42)
    main()
