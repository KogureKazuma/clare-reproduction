"""1-D CNN with late fusion for multimodal cognitive load classification.

Architecture (per modality branch):
  4 VGG-style blocks, each: Conv1d → ReLU → Conv1d → ReLU → MaxPool1d
  After 4 blocks: AdaptiveAvgPool → Flatten → FC(128)

Late fusion:
  Concatenate branch outputs → FC(256) → FC(2)

Training: AdaDelta (rho=0.95, lr=5e-3), Focal Loss (alpha=4.0, gamma=2.0)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 4.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


class ConvBlock(nn.Module):
    """Two Conv1d layers with ReLU, followed by MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModalityBranch(nn.Module):
    """4-block CNN branch for a single modality."""

    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()
        channels = [32, 64, 128, 256]
        layers = [ConvBlock(in_channels, channels[0])]
        for i in range(1, 4):
            layers.append(ConvBlock(channels[i - 1], channels[i]))
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.cnn(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class MultiModalCNN(nn.Module):
    """Late-fusion multimodal 1D CNN."""

    def __init__(
        self,
        modality_channels: dict[str, int],
        embed_dim: int = 128,
        n_classes: int = 2,
    ):
        """
        Args:
            modality_channels: dict mapping modality name → number of input channels
            embed_dim: per-branch embedding size
            n_classes: number of output classes
        """
        super().__init__()
        self.modality_names = list(modality_channels.keys())
        self.branches = nn.ModuleDict({
            name: ModalityBranch(n_ch, embed_dim)
            for name, n_ch in modality_channels.items()
        })
        fused_dim = embed_dim * len(modality_channels)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_dict: modality name → (B, n_channels, T) tensor

        Returns:
            (B, n_classes) logits
        """
        embeds = [self.branches[name](x_dict[name]) for name in self.modality_names]
        fused = torch.cat(embeds, dim=-1)
        return self.head(fused)


def build_cnn(modality_channels: dict[str, int], n_classes: int = 2) -> MultiModalCNN:
    return MultiModalCNN(modality_channels, embed_dim=128, n_classes=n_classes)
