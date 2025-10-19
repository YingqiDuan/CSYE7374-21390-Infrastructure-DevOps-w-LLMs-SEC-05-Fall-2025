"""
Utilities for loading the pre-processed dataset from Assignment 1 and preparing
PyTorch dataloaders for the mini-GPT training script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


@dataclass(slots=True)
class DatasetSummary:
    """Lightweight container capturing basic dataset statistics."""

    num_sequences: int
    vocab_size: int
    max_seq_len: int


class TokenSequenceDataset(Dataset):
    """
    Dataset that stores token sequences along with their attention masks.

    Each item represents a single contiguous sequence (unbatched) where the
    attention mask indicates valid tokens (value 1) and padding (value 0).
    """

    def __init__(self, sequences: Sequence[torch.Tensor], attention_masks: Sequence[torch.Tensor]) -> None:
        if len(sequences) != len(attention_masks):
            raise ValueError("Sequences and attention_masks must have identical length")
        self.sequences = tuple(sequences)
        self.attention_masks = tuple(attention_masks)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.sequences[idx],
            "attention_mask": self.attention_masks[idx],
        }


def _flatten_assignment1_batches(batches: Iterable[dict[str, torch.Tensor]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert the saved mini-batches from Assignment 1 into individual sequences."""
    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    for batch in batches:
        input_ids = batch["input_ids"]
        masks = batch["attention_mask"]
        if input_ids.ndim != 2 or masks.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors for input_ids and attention_mask, got shapes "
                f"{tuple(input_ids.shape)} and {tuple(masks.shape)}."
            )
        if input_ids.shape != masks.shape:
            raise ValueError("input_ids and attention_mask must share the same shape")
        valid_lengths = masks.sum(dim=1).tolist()
        for seq, mask, valid_length in zip(input_ids, masks, valid_lengths, strict=True):
            if valid_length < 2:
                continue  # Need at least two tokens for next-token prediction.
            sequences.append(seq[:valid_length].clone())
            attention_masks.append(mask[:valid_length].clone())
    return sequences, attention_masks


def load_token_dataset(path: str | Path) -> tuple[TokenSequenceDataset, DatasetSummary]:
    """
    Load the saved dataset (list of batches) exported during Assignment 1.

    Args:
        path: Path to the `.pt` file produced by `save_sample_batches`.

    Returns:
        A dataset ready for DataLoader consumption and summary statistics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file {path} does not exist.")

    batches = torch.load(path, map_location="cpu")
    if not isinstance(batches, (list, tuple)):
        raise TypeError("Expected the dataset file to contain a list/tuple of batches.")

    sequences, masks = _flatten_assignment1_batches(batches)
    if not sequences:
        raise ValueError("Loaded dataset is empty.")

    max_token_id = max(seq.max().item() for seq in sequences)
    max_seq_len = max(seq.size(0) for seq in sequences)
    dataset = TokenSequenceDataset(sequences, masks)
    summary = DatasetSummary(
        num_sequences=len(sequences),
        vocab_size=max_token_id + 1,
        max_seq_len=max_seq_len,
    )
    return dataset, summary


def pad_collate_fn(batch: Sequence[dict[str, torch.Tensor]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """
    Collate function that pads variable-length sequences within a batch.

    Returns:
        Dictionary with padded `input_ids`, `attention_mask`, and `targets`
        (identical to `input_ids`, not shifted).
    """
    if not batch:
        raise ValueError("Received empty batch in collate_fn.")

    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]

    first_device = input_ids[0].device
    if any(t.device != first_device for t in (*input_ids, *attention_masks)):
        raise ValueError("All tensors passed to pad_collate_fn must reside on the same device.")

    if __debug__:
        for idx, mask in enumerate(attention_masks):
            if not torch.all((mask == 0) | (mask == 1)).item():
                raise ValueError(f"attention_mask at index {idx} must contain only 0/1 values.")

    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    if padded_inputs.dtype != torch.long:
        padded_inputs = padded_inputs.long()

    padded_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    if padded_masks.dtype != torch.long:
        padded_masks = padded_masks.long()

    targets = padded_inputs.clone()
    return {
        "input_ids": padded_inputs,
        "attention_mask": padded_masks,
        "targets": targets,
    }
