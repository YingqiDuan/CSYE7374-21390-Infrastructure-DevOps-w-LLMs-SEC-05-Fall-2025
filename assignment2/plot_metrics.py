"""
Utility script to visualise loss and perplexity curves from `training_log.json`.

Example:
    python plot_metrics.py --log-file runs/exp01/training_log.json --output runs/exp01/metrics.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss and perplexity curves.")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to training_log.json produced during training.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path. Defaults to same folder as log.")
    return parser.parse_args()


def _validate_history_entry(entry: dict[str, Any], index: int) -> None:
    required_keys = ["epoch", "train_loss", "val_loss", "train_perplexity", "val_perplexity"]
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise KeyError(f"History entry #{index} is missing keys: {', '.join(missing)}")


def _extract_metric(history: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(entry[key]) for entry in history]


def main() -> None:
    args = parse_args()
    log_path: Path = args.log_file
    if not log_path.exists():
        raise FileNotFoundError(f"Log file {log_path} not found.")

    with log_path.open("r", encoding="utf-8") as fh:
        history_data: Any = json.load(fh)

    if not isinstance(history_data, list):
        raise TypeError("Expected training history to be a list of dict entries.")
    history: list[dict[str, Any]] = history_data
    if not history:
        raise ValueError("History is empty; nothing to plot.")

    # Ensure entries are validated and sorted by epoch for reproducible curves.
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise TypeError(f"History entry #{idx} is not a JSON object.")
        _validate_history_entry(entry, idx)
    history.sort(key=lambda x: x["epoch"])

    epochs = [int(entry["epoch"]) for entry in history]
    train_loss = _extract_metric(history, "train_loss")
    val_loss = _extract_metric(history, "val_loss")
    train_ppl = _extract_metric(history, "train_perplexity")
    val_ppl = _extract_metric(history, "val_perplexity")

    def insert_nan(values: Sequence[float]) -> list[float]:
        return [value if math.isfinite(value) else float("nan") for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_loss, ax_ppl = axes

    loss_y1 = insert_nan(train_loss)
    loss_y2 = insert_nan(val_loss)
    ax_loss.plot(epochs, loss_y1, marker="o", label="Train loss")
    ax_loss.plot(epochs, loss_y2, marker="s", label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    run_name = log_path.parent.name or log_path.stem
    ax_loss.set_title(f"Training vs Validation Loss – {run_name}")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xticks(epochs)
    ax_loss.legend()

    ppl_y1 = insert_nan(train_ppl)
    ppl_y2 = insert_nan(val_ppl)
    ax_ppl.plot(epochs, ppl_y1, marker="o", label="Train perplexity")
    ax_ppl.plot(epochs, ppl_y2, marker="s", label="Val perplexity")
    ax_ppl.set_xlabel("Epoch")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.set_title(f"Training vs Validation Perplexity – {run_name}")
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.set_xticks(epochs)
    ax_ppl.legend()

    plt.tight_layout()
    output_path: Path = args.output if args.output else log_path.with_name("metrics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plots to {output_path}")


if __name__ == "__main__":
    main()
