import os
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history: list[float], output_dir: str = "output") -> None:
    """
    Plot the training loss over iterations and save it as an image.

    Args:
        loss_history: list of average loss values recorded each epoch
        output_dir  : directory where the image will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    iterations = list(range(1, len(loss_history) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        iterations, loss_history, color="#2196F3", linewidth=1.8, label="Training Loss"
    )

    # Mark the final loss value on the plot
    final_loss = loss_history[-1]
    ax.annotate(
        f"Final: {final_loss:.4f}",
        xy=(iterations[-1], final_loss),
        xytext=(-80, 15),
        textcoords="offset points",
        fontsize=9,
        color="#E53935",
        arrowprops=dict(arrowstyle="->", color="#E53935"),
    )

    ax.set_title(
        "Mirror Mirror — Training Loss vs. Iteration", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[display] {save_path}")
