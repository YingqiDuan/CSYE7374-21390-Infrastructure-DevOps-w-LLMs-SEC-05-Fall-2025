"""
Training script for a compact GPT-style language model on the Assignment 1 dataset.

Usage example:
    python train_mini_gpt.py \\
        --dataset ../assignment1/sample_dataset.pt \\
        --output-dir runs/exp01 \\
        --epochs 3 --batch-size 8 --embed-dim 128 --num-layers 2 --num-heads 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import sys

from dataclasses import asdict

import torch
import torch.nn.functional as F
from functools import partial

from torch.amp import GradScaler, autocast
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader, Dataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from data_utils import DatasetSummary, load_token_dataset, pad_collate_fn
from mini_gpt import MiniGPT, MiniGPTConfig


class _SequenceTruncationView(Dataset):
    """View over a sequence dataset that truncates tokens to `max_length` without copying."""

    def __init__(self, base_dataset: Dataset, max_length: int) -> None:
        self._base_dataset = base_dataset
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._base_dataset[idx]
        return {
            "input_ids": item["input_ids"][: self._max_length],
            "attention_mask": item["attention_mask"][: self._max_length],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None:
        np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mini GPT model on the processed dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("../assignment1/sample_dataset.pt"), help="Path to the tokenised dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"), help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Optional cap on sequence length.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data reserved for validation.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    default_workers = max(1, cpu_count() // 2)
    parser.add_argument("--num-workers", type=int, default=default_workers, help="Number of DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training log prints.")
    parser.add_argument("--save-every", type=int, default=None, help="Optional steps frequency for intermediate checkpoints.")
    args = parser.parse_args()
    if not (0.0 < args.val_split < 1.0):
        parser.error("--val-split must be within the open interval (0, 1).")
    return args


def prepare_dataloaders(
    dataset_path: Path,
    batch_size: int,
    val_split: float,
    max_seq_len_override: int | None,
    seed: int,
    pin_memory: bool,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DatasetSummary]:
    dataset, summary = load_token_dataset(dataset_path)
    if max_seq_len_override is not None:
        summary.max_seq_len = min(summary.max_seq_len, max_seq_len_override)
        dataset = _SequenceTruncationView(dataset, summary.max_seq_len)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Validation split ({val_split}) is too large for dataset of size {len(dataset)} "
            f"(train would be {train_size}, val {val_size})."
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    collate_fn = partial(pad_collate_fn, pad_token_id=0, target_length=summary.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader, summary


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for next-token prediction, ignoring padding tokens.
    """
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_targets.reshape(-1),
        reduction="none",
    )
    loss = loss * shift_mask.reshape(-1)
    denom = shift_mask.sum()
    if denom.item() == 0:
        return loss.mean()
    return loss.sum() / denom


def evaluate(
    model: MiniGPT,
    dataloader: DataLoader,
    device: torch.device,
    non_blocking: bool,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)
            with autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = compute_loss(logits, targets, attention_mask)
            valid_tokens = attention_mask[:, 1:].sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    mean_loss = total_loss / max(total_tokens, 1.0)
    perplexity = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    return mean_loss, perplexity


def save_checkpoint(
    output_dir: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float]],
    global_step: int,
    filename: str | None = None,
    extra: dict[str, float | int | bool | list | dict] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": asdict(model.config),
        "global_step": global_step,
    }
    if extra:
        payload.update(extra)
    ckpt_name = filename or f"mini_gpt_epoch{epoch:03d}.pt"
    ckpt_path = output_dir / ckpt_name
    torch.save(payload, ckpt_path)
    return ckpt_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_log.json"

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except AttributeError:
            pass
    train_loader, val_loader, summary = prepare_dataloaders(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_seq_len_override=args.max_seq_len,
        seed=args.seed,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    steps_per_epoch = train_size // args.batch_size
    print(
        f"Dataset split: train={train_size} validation={val_size} | "
        f"steps per epoch (drop_last=True)={steps_per_epoch}"
    )

    config = MiniGPTConfig(
        vocab_size=summary.vocab_size,
        max_seq_len=summary.max_seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    global_step = 0
    epoch = 0
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_tokens = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
                attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
                targets = batch["targets"].to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = compute_loss(logits, targets, attention_mask)

                if use_amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip is not None and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip is not None and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                valid_tokens = attention_mask[:, 1:].sum().item()
                running_loss += loss.item() * valid_tokens
                running_tokens += valid_tokens
                global_step += 1

                if global_step % args.log_interval == 0:
                    mean_loss = running_loss / max(running_tokens, 1.0)
                    perplexity = math.exp(mean_loss) if mean_loss < 50 else float("inf")
                    print(
                        f"[Epoch {epoch}/{args.epochs}] step {global_step} | "
                        f"tokens {int(running_tokens)} | loss {mean_loss:.4f} | ppl {perplexity:.2f}"
                    )
                    running_loss = 0.0
                    running_tokens = 0.0

                if args.save_every and global_step % args.save_every == 0:
                    save_checkpoint(output_dir, model, optimizer, epoch, history, global_step)

            train_loss, train_ppl = evaluate(model, train_loader, device, pin_memory, use_amp)
            val_loss, val_ppl = evaluate(model, val_loader, device, pin_memory, use_amp)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_perplexity": train_ppl,
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                }
            )
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
                f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
            )
            save_checkpoint(output_dir, model, optimizer, epoch, history, global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    epoch,
                    history,
                    global_step,
                    filename="mini_gpt_best.pt",
                    extra={"best_val_loss": best_val_loss},
                )

            with history_path.open("w", encoding="utf-8") as fp:
                json.dump(history, fp, indent=2)

    except KeyboardInterrupt:
        last_path = save_checkpoint(
            output_dir,
            model,
            optimizer,
            epoch,
            history,
            global_step,
            filename="mini_gpt_last.pt",
            extra={"interrupted": True},
        )
        with history_path.open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        print(f"Training interrupted. Last checkpoint saved to {last_path}.")
        raise

    else:
        save_checkpoint(
            output_dir,
            model,
            optimizer,
            epoch,
            history,
            global_step,
            filename="mini_gpt_last.pt",
            extra={"interrupted": False, "best_val_loss": best_val_loss},
        )


if __name__ == "__main__":
    main()
