import numpy as np
from data.data import generate_data, make_demo_sequence, make_clean_sequence, MIDDLE
from core.brnn_model import BidirectionalRNN

# Configuration
BAR = 28
NUM_EPOCHS = 10
LR = 0.005


def loss_bar(loss: float, max_loss: float = 0.05) -> str:
    filled = int(BAR * min(loss / max_loss, 1.0))
    return "█" * filled + "░" * (BAR - filled)


def predict_clean(model: BidirectionalRNN, seq_input: list) -> float:
    outputs = model.feedforward(seq_input)
    return float(outputs[MIDDLE][0, 0])


def train_model(model, train_data, demo_input, demo_true, demo_max):
    print("\n" + "━" * 100)
    print("  TRAINING")
    print("━" * 100)
    print("  The model sees a corrupted sequence and learns to")
    print("  predict what the clean middle value should be.")
    print("  (* marks the corrupted value)\n")
    print(f"  Demo sequence : {make_demo_sequence(2, 2.0)[2]}")
    print(
        f"  Clean middle  : {demo_true * demo_max:.2f}  (normalized: {demo_true:.4f})\n"
    )
    print(f"  {'Epoch':<7} {'Loss':<12} {'Bar':<{BAR}}  {'Demo prediction'}")
    print(f"  {'─'*7} {'─'*12} {'─'*BAR}  {'─'*28}")

    full_inputs, full_targets = train_data

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0

        for i in range(len(full_inputs)):
            outputs = model.feedforward(full_inputs[i])
            model.backpropagation(full_inputs[i], outputs, full_targets[i], LR)
            epoch_loss += model.loss(outputs, full_targets[i])

        epoch_loss /= len(full_inputs)
        pred = predict_clean(model, demo_input)
        error = abs(pred - demo_true)
        demo = f"pred={pred * demo_max:.2f}  true={demo_true * demo_max:.2f}  err={error:.4f}"
        print(f"  {epoch:<7} {epoch_loss:<12.6f} {loss_bar(epoch_loss)}  {demo}")

    print()


def evaluate_model(model, test_data):
    print("━" * 100)
    print("  EVALUATION")
    print("━" * 100)

    test_inputs, test_targets = test_data
    test_loss = np.mean(
        [model.loss(model.feedforward(x), y) for x, y in zip(test_inputs, test_targets)]
    )
    print(f"\n  Test loss (MSE): {test_loss:.6f}\n")
    print(f"  {'Corrupted sequence':<32} {'Predicted':>10} {'True':>10} {'Error':>8}")
    print(f"  {'─'*32} {'─'*10} {'─'*10} {'─'*8}")

    checks = [(2, 1.5), (3, 2.0), (5, 1.8), (1, 2.5), (4, 1.6)]
    for start, ratio in checks:
        seq_in, true_val, display, max_val = make_demo_sequence(start, ratio)
        pred = predict_clean(model, seq_in)
        error = abs(pred - true_val)
        print(
            f"  {display:<32} {pred * max_val:>10.2f} {true_val * max_val:>10.2f} {error:>8.4f}"
        )

    print()


def interactive_demo(model):
    print("━" * 100)
    print("  PREDICT  (type 'quit' to exit)")
    print("━" * 100)
    print("  Enter a start value (1-10) and ratio (1.5-5.0).")
    print("  The model will correct the corrupted middle value.\n")
    print("  Example: '2 3' → clean sequence is  2.00  6.00  18.00  54.00  162.00")
    print("           The middle (18.00) gets corrupted → model predicts ~18.00\n")

    while True:
        try:
            raw = input("  start  ratio  (e.g. '2 2.0')  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Okie!")
            break

        if raw in ("quit", "exit", "q"):
            print("  Bye!")
            break

        if not raw:
            continue

        try:
            parts = raw.replace(",", " ").split()
            start = int(parts[0])
            ratio = float(parts[1])
            if start <= 0 or start > 10 or ratio < 1.5 or ratio > 5.0:
                print("  Invalid input. Please follow: (start: 1-10, ratio: 1.5-5.0)\n")
                continue
        except (ValueError, IndexError):
            print("  (enter a positive integer and a ratio > 1.0, e.g. '2 2')\n")
            continue

        seq_in, true_val, display, max_val = make_demo_sequence(start, ratio)
        pred = predict_clean(model, seq_in)
        pred_raw = pred * max_val
        true_raw = true_val * max_val
        error = abs(pred_raw - true_raw)
        clean_raw = make_clean_sequence(start, ratio)
        clean_str = "  ".join(f"{v:.2f}" for v in clean_raw)

        print()
        print(f"  Clean sequence    : {clean_str}")
        print(f"  Corrupted input   : {display}  (* = corrupted value)")
        print(f"  Model prediction  : {pred_raw:.2f}")
        print(f"  True clean value  : {true_raw:.2f}")
        print(f"  Error             : {error:.2f}")
        print()


def main():
    # Data preparation
    train_data = generate_data(num_samples=1000)
    test_data = generate_data(num_samples=200)

    # Demo sequence
    demo_input, demo_true, demo_str, demo_max = make_demo_sequence(2, 2.0)

    # Model
    model = BidirectionalRNN(1, 16, 1)

    # Run
    train_model(model, train_data, demo_input, demo_true, demo_max)
    evaluate_model(model, test_data)
    interactive_demo(model)


if __name__ == "__main__":
    main()
