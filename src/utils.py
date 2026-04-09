"""
utils.py — Shared utilities for the TESS exoplanet detection pipeline.

Provides: seed management, Focal Loss, metric computation, config loading.
"""
from __future__ import annotations

import random
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a nested dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Loss ──────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss — Lin et al., 2017.

    Focuses training on hard-to-classify examples by down-weighting easy
    negatives. Critical for our severely imbalanced dataset (~2-5% positive).

    Args:
        alpha: Weight for the positive class. Set > 0.5 to up-weight planets.
        gamma: Focusing parameter. gamma=0 reduces to BCE; gamma=2 is typical.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "threshold": threshold,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_val, best_thresh = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            val = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        if val > best_val:
            best_val, best_thresh = val, t
    return best_thresh


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(
    state: dict,
    path: str | Path,
    is_best: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "best_model.pt"
        torch.save(state, best_path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)