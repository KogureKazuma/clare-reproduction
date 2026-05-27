"""Transformer with early fusion for multimodal cognitive load classification.

Architecture:
  Early fusion: interpolate/pad each modality to a common sequence length,
                then concatenate along the channel axis.
  4 Transformer encoder blocks (multi-head attention + FFN)
  Global average pooling → FC(256) → FC(128) → FC(n_classes)

Training: Adam (lr=1e-4), batch=256, epochs=100
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class MultiModalTransformer(nn.Module):
    """Early-fusion Transformer for multimodal time-series classification."""

    def __init__(
        self,
        modality_channels: dict[str, int],
        seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        """
        Args:
            modality_channels: modality name → number of raw channels
            seq_len: unified sequence length (signals resampled to this)
            d_model: transformer model dimension
        """
        super().__init__()
        self.modality_names = list(modality_channels.keys())
        self.seq_len = seq_len

        total_channels = sum(modality_channels.values())

        # project concatenated channels to d_model
        self.input_proj = nn.Linear(total_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        """Resample signal (B, C, T) → (B, C, seq_len)."""
        if x.shape[-1] == self.seq_len:
            return x
        return F.interpolate(x.float(), size=self.seq_len, mode="linear", align_corners=False)

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_dict: modality name → (B, n_channels, T) tensor

        Returns:
            (B, n_classes) logits
        """
        resampled = [
            self._resample(x_dict[name]) for name in self.modality_names
        ]
        # (B, total_channels, seq_len) → (B, seq_len, total_channels)
        fused = torch.cat(resampled, dim=1).permute(0, 2, 1)

        h = self.input_proj(fused)
        h = self.pos_enc(h)

        for block in self.blocks:
            h = block(h)

        # global average pool over time
        h = h.mean(dim=1)
        return self.head(h)


def build_transformer(
    modality_channels: dict[str, int],
    seq_len: int = 128,
    n_classes: int = 2,
) -> MultiModalTransformer:
    return MultiModalTransformer(
        modality_channels=modality_channels,
        seq_len=seq_len,
        d_model=128,
        n_heads=4,
        n_blocks=4,
        ffn_dim=256,
        n_classes=n_classes,
    )
