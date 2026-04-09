"""
train.py — Training loop for the DualViewTransformer.

Features:
  - Focal Loss for class imbalance
  - Cosine annealing LR with linear warmup
  - Stratified train/val/test split
  - Early stopping on validation ROC-AUC
  - Weights & Biases logging (optional)
  - Checkpoint saving (best + latest)

Usage:
    python src/train.py --config configs/config.yaml [--baseline]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_baseline, build_model, count_parameters
from src.utils import (
    FocalLoss,
    compute_metrics,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────
class TESSDataset(Dataset):
    def __init__(
        self,
        global_views: np.ndarray,
        local_views: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.05,
    ) -> None:
        assert len(global_views) == len(local_views) == len(labels)
        self.global_views = torch.tensor(global_views, dtype=torch.float32)
        self.local_views  = torch.tensor(local_views,  dtype=torch.float32)
        self.labels       = torch.tensor(labels,       dtype=torch.float32)
        self.augment  = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gv = self.global_views[idx].clone()
        lv = self.local_views[idx].clone()
        label = self.labels[idx]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                gv = gv.flip(0)
                lv = lv.flip(0)
            gv = gv + torch.randn_like(gv) * self.noise_std
            lv = lv + torch.randn_like(lv) * self.noise_std
        return gv, lv, label


# ── Data loading helpers ──────────────────────────────────────────────────────
def load_processed_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    processed = Path(config["paths"]["processed_dir"])
    global_views = np.load(processed / "global_views.npy")
    local_views  = np.load(processed / "local_views.npy")
    labels       = np.load(processed / "labels.npy")
    return global_views, local_views, labels


def make_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    cfg = config["data"]
    global_views, local_views, labels = load_processed_data(config)

    idx = np.arange(len(labels))
    idx_train, idx_tmp, _, y_tmp = train_test_split(
        idx, labels,
        test_size=cfg["val_fraction"] + cfg["test_fraction"],
        stratify=labels, random_state=cfg["seed"],
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, stratify=y_tmp, random_state=cfg["seed"],
    )
    logger.info(f"Split → train: {len(idx_train):,}  val: {len(idx_val):,}  test: {len(idx_test):,}")

    batch = config["training"]["batch_size"]
    train_ds = TESSDataset(global_views[idx_train], local_views[idx_train], labels[idx_train], augment=True)
    val_ds   = TESSDataset(global_views[idx_val],   local_views[idx_val],   labels[idx_val])
    test_ds  = TESSDataset(global_views[idx_test],  local_views[idx_test],  labels[idx_test])

    class_counts = np.bincount(labels[idx_train].astype(int))
    weights = 1.0 / class_counts[labels[idx_train].astype(int)]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch * 2, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch * 2, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_scheduler(optimizer: optim.Optimizer, config: dict, steps_per_epoch: int):
    cfg = config["training"]
    total_steps  = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Evaluation pass ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, dict]:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for gv, lv, labels in loader:
        gv, lv, labels = gv.to(device), lv.to(device), labels.to(device)
        logits, _ = model(gv, lv)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ── Main training loop ────────────────────────────────────────────────────────
def train(config: dict, use_baseline: bool = False) -> None:
    set_seed(config["data"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    use_wandb = config["logging"]["use_wandb"]
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"]["wandb_entity"],
                config=config,
                name="baseline-cnn" if use_baseline else "dual-view-transformer",
            )
        except ImportError:
            logger.warning("wandb not installed — skipping logging.")
            use_wandb = False

    if use_baseline:
        model = build_baseline(config, device)
        logger.info("Training BASELINE 1D-CNN.")
    else:
        model = build_model(config, device)
        logger.info("Training DUAL-VIEW TRANSFORMER.")
    logger.info(f"Trainable parameters: {count_parameters(model):,}")

    train_loader, val_loader, test_loader = make_dataloaders(config)

    cfg_t     = config["training"]
    criterion = FocalLoss(alpha=cfg_t["focal_alpha"], gamma=cfg_t["focal_gamma"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg_t["learning_rate"], weight_decay=cfg_t["weight_decay"])
    scheduler = get_scheduler(optimizer, config, len(train_loader))

    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    epochs_without_improvement = 0
    log_interval = config["logging"]["log_interval"]

    for epoch in range(1, cfg_t["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for step, (gv, lv, labels) in enumerate(train_loader, 1):
            gv, lv, labels = gv.to(device), lv.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(gv, lv)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_t["grad_clip"])
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            if step % log_interval == 0:
                logger.debug(f"Epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        avg_train_loss = epoch_loss / len(train_loader)
        auc = val_metrics["roc_auc"]

        logger.info(
            f"Epoch {epoch:3d} | train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f} "
            f"| AUC={auc:.4f}  F1={val_metrics['f1']:.4f}  Recall={val_metrics['recall']:.4f}"
        )

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss,
                       "lr": scheduler.get_last_lr()[0], **{f"val_{k}": v for k, v in val_metrics.items()}})

        state = {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "val_metrics": val_metrics, "config": config}
        is_best = auc > best_auc
        save_checkpoint(state, ckpt_dir / "latest.pt", is_best=is_best)
        if is_best:
            best_auc = auc
            epochs_without_improvement = 0
            logger.info(f"  ✓ New best AUC: {best_auc:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg_t["patience"]:
            logger.info(f"Early stopping after {epoch} epochs.")
            break

    logger.info("Loading best checkpoint for final test evaluation...")
    best_state = load_checkpoint(ckpt_dir / "best_model.pt", device)
    model.load_state_dict(best_state["model_state"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"\n{'='*50}\nFINAL TEST RESULTS\n{'='*50}\n"
                + "\n".join(f"  {k}: {v:.4f}" for k, v in test_metrics.items()))

    if use_wandb:
        import wandb
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, use_baseline=args.baseline)