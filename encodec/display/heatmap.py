import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_hidden_state_heatmap(
    encoder_step_cache: list,
    input_digits: list[int],
    output_dir: str = "output",
) -> None:
    """
    Visualize how the encoder's hidden state evolves as it reads the
    input sequence, saved as a heatmap image.

    Rows    = hidden units (neurons)
    Columns = time steps (one per input token)

    Args:
        encoder_step_cache: list of per-step cache dicts from the encoder's
                            forward pass (each contains 'hidden_state')
        input_digits       : list of digit integers from the input sequence
                             (used as column labels)
        output_dir         : directory where the image will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build matrix: shape (hidden_size, sequence_length)
    hidden_matrix = np.hstack([cache["hidden_state"] for cache in encoder_step_cache])

    hidden_size, sequence_length = hidden_matrix.shape

    fig, ax = plt.subplots(figsize=(sequence_length * 1.5 + 2, 6))

    heatmap = ax.imshow(
        hidden_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # Colorbar to show activation magnitude
    colorbar = plt.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.04)
    colorbar.set_label("Activation (tanh)", fontsize=10)

    # Column labels: show each input digit with its time step index
    column_labels = [f"t={t}\n('{d}')" for t, d in enumerate(input_digits)]
    ax.set_xticks(range(sequence_length))
    ax.set_xticklabels(column_labels, fontsize=9)

    # Row labels: show every 8th hidden unit index to avoid clutter
    ax.yaxis.set_major_locator(ticker.MultipleLocator(max(1, hidden_size // 8)))
    ax.set_ylabel("Hidden Unit Index", fontsize=11)
    ax.set_xlabel("Encoder Time Step", fontsize=11)
    reversed_digits = list(reversed(input_digits))
    ax.set_title(
        f"Mirror Mirror — Encoder Hidden State Heatmap\n"
        f"Input: {input_digits}   →   Expected Output: {reversed_digits}",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, "hidden_state_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[display] {save_path}")
