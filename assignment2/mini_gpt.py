"""
Minimal GPT-style transformer model for next-token prediction.

This module defines a compact decoder-only transformer with configurable
depth, attention heads, embedding size, and dropout. It relies only on core
PyTorch primitives so it can run in constrained environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class MiniGPTConfig:
    """Configuration container for the MiniGPT model."""

    vocab_size: int
    max_seq_len: int
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    feedforward_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.feedforward_dim is None:
            # Follow GPT-2 convention: 4x hidden size for the MLP projection.
            self.feedforward_dim = 4 * self.embed_dim


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return an upper-triangular mask to enforce auto-regressive attention."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm residual structure."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class MiniGPT(nn.Module):
    """Decoder-only transformer for next-token prediction."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.register_buffer("_position_ids", torch.arange(config.max_seq_len), persistent=False)
        self.register_buffer("_causal_mask", build_causal_mask(config.max_seq_len), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Binary mask of same shape where 1 indicates valid tokens. (Currently ignored.)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.size()
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len={self.config.max_seq_len}"
            )

        pos_embeddings = self.pos_embed(self._position_ids[:seq_len])[None, :, :]
        token_embeddings = self.token_embed(input_ids)
        x = self.dropout(token_embeddings + pos_embeddings)

        # Slice the cached causal mask for the current sequence length.
        causal_mask = self._causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation utility for qualitative evaluation."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                generated = input_ids
                for _ in range(max_new_tokens):
                    logits = self.forward(generated[:, -self.config.max_seq_len :])
                    next_logits = logits[:, -1, :] / max(temperature, 1e-6)
                    if top_k is not None:
                        values, _ = torch.topk(next_logits, top_k)
                        min_values = values[:, -1].unsqueeze(1)
                        next_logits = torch.where(next_logits < min_values, torch.full_like(next_logits, float("-inf")), next_logits)
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
        finally:
            if was_training:
                self.train()
            else:
                self.eval()
        return generated
