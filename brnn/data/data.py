import numpy as np

# Each sequence has 5 terms, position 0 to 4
# The corrupted value is always at position 2 (the middle)
SEQUENCE_LENGTH = 5
MIDDLE = SEQUENCE_LENGTH // 2  # = 2

# How much noise we add to corrupt the middle value
# Too small = too easy, too large = impossible to recover
NOISE_LEVEL = 0.3


def make_clean_sequence(
    start: int, ratio: float, length: int = SEQUENCE_LENGTH
) -> list:
    # Generate a clean geometric sequence
    # Example: start=2, ratio=2 → [2, 4, 8, 16, 32]
    return [start * (ratio**i) for i in range(length)]


def normalize(sequence: list) -> list:
    # Scale all values so the max is 1.0
    # Keeps the model working with small numbers (avoids large gradients)
    max_val = max(sequence)
    return [x / max_val for x in sequence]


def add_noise(value: float, noise_level: float = NOISE_LEVEL) -> float:
    # Corrupt a value by adding random noise
    # Noise is sampled from [-noise_level, +noise_level]
    return value + np.random.uniform(-noise_level, noise_level)


def generate_data(num_samples: int = 1000) -> tuple:
    all_inputs = []
    all_targets = []

    for _ in range(num_samples):
        # Step 1: Random start (1-10) and ratio (1.5-5.0)
        start = np.random.randint(1, 11)  # 1-10
        ratio = np.random.uniform(1.5, 5.0)  # 1.5-5.0

        # Step 2: Generate and normalize the clean sequence
        clean = normalize(make_clean_sequence(start, ratio))

        # Step 3: Corrupt the middle value with noise
        corrupted = clean.copy()
        corrupted[MIDDLE] = add_noise(clean[MIDDLE])

        # Step 4: Wrap each value into shape (1, 1) for the RNN
        seq_input = [np.array([[x]]) for x in corrupted]

        # Step 5: Target is just the clean middle value
        seq_target = [np.array([[clean[MIDDLE]]])]

        all_inputs.append(seq_input)
        all_targets.append(seq_target)

    return all_inputs, all_targets


def make_demo_sequence(start: int, ratio: float, noise: float = None) -> tuple:
    # Build one sequence for display purposes
    # If noise is None, use the default NOISE_LEVEL
    if noise is None:
        noise = NOISE_LEVEL

    clean = normalize(make_clean_sequence(start, ratio))

    # Corrupt the middle
    corrupted = clean.copy()
    corrupted[MIDDLE] = add_noise(clean[MIDDLE], noise)

    # Un-normalized raw values for display
    raw_clean = make_clean_sequence(start, ratio)
    max_val = max(raw_clean)

    # Build a readable display string
    display_parts = []
    for i, v in enumerate(raw_clean):
        if i == MIDDLE:
            corrupted_raw = corrupted[MIDDLE] * max_val
            display_parts.append(f"{corrupted_raw:.2f}*")  # Marks the corrupted
        else:
            display_parts.append(f"{v:.2f}")
    display_str = "  ".join(display_parts)

    seq_input = [np.array([[x]]) for x in corrupted]
    true_val = clean[MIDDLE]

    return seq_input, true_val, display_str, max_val
