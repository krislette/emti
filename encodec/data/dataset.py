import numpy as np


# Vocabulary: digits 0-9
VOCABULARY_SIZE = 10
SEQUENCE_LENGTH = 5


def generate_sequences(num_samples: int) -> list[tuple[list[int], list[int]]]:
    """
    Generate random digit sequences and their reversals.

    Each sample is a tuple of:
      - input_sequence : list of ints, e.g. [4, 1, 9, 2, 7]
      - target_sequence: list of ints, e.g. [7, 2, 9, 1, 4]
    """
    dataset = []

    for _ in range(num_samples):
        input_sequence = np.random.randint(
            0, VOCABULARY_SIZE, size=SEQUENCE_LENGTH
        ).tolist()
        target_sequence = list(reversed(input_sequence))
        dataset.append((input_sequence, target_sequence))

    return dataset


def one_hot_encode(digit: int, vocab_size: int = VOCABULARY_SIZE) -> np.ndarray:
    """
    Convert a single digit into a one-hot column vector.

    Example: digit=3, vocab_size=10 ->
      [[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]]

    Shape: (vocab_size, 1)
    """
    vector = np.zeros((vocab_size, 1))
    vector[digit] = 1.0
    return vector


def encode_sequence(sequence: list[int]) -> list[np.ndarray]:
    """
    Convert a list of digits into a list of one-hot column vectors.

    Shape per vector: (VOCABULARY_SIZE, 1)
    """
    return [one_hot_encode(digit) for digit in sequence]


def decode_prediction(output_vector: np.ndarray) -> int:
    """
    Convert a raw output vector (logits or softmax) back to a digit
    by taking the argmax.

    Input shape: (vocab_size, 1)
    """
    return int(np.argmax(output_vector))
