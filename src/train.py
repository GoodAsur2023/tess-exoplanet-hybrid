"""
train.py — Training loop for the DualViewTransformer Ablation Study.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Add project root to path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import DualViewTransformer
from src.utils import (
    get_device,
    load_config,
    set_seed,
    compute_metrics
)

# --- HELPER FUNCTION ---
def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Focal Loss (Redefined here for safety out-of-the-box)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        return focal_loss.mean()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset ───────────────────────────────────────────────────────────────────
class TESSDataset(Dataset):
    def __init__(self, global_views, local_views, stellar_meta, labels, fusion_type="mlp"):
        self.gv = torch.tensor(global_views, dtype=torch.float32).unsqueeze(1)
        self.lv = torch.tensor(local_views, dtype=torch.float32).unsqueeze(1)
        self.meta = torch.tensor(stellar_meta, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.fusion_type = fusion_type

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gv_i = self.gv[idx]
        lv_i = self.lv[idx]
        meta_i = self.meta[idx]
        
        # === ABLATION 4: The Science Way (Astrophysics) ===
        if self.fusion_type == "astrophysics":
            # meta_i[2] is the normalized stellar radius. 
            # We explicitly remove the stellar radius bias from the flux dip.
            rad_squared = (meta_i[2] ** 2)
            gv_i = gv_i * rad_squared
            lv_i = lv_i * rad_squared

        return gv_i, lv_i, meta_i, self.labels[idx]

# ── Dataloaders ───────────────────────────────────────────────────────────────
def make_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    cfg_paths = config["paths"]
    cfg_data = config["data"]
    cfg_train = config["training"]
    fusion_type = config["model"].get("fusion_type", "mlp")

    data_dir = Path(cfg_paths["processed_dir"])
    logger.info(f"Loading arrays from: {data_dir}")
    gv = np.load(data_dir / "global_views.npy")
    lv = np.load(data_dir / "local_views.npy")
    mt = np.load(data_dir / "stellar_meta.npy")
    lb = np.load(data_dir / "labels.npy")

    # Stratified Split (Train / Temp)
    test_val_ratio = cfg_data["val_fraction"] + cfg_data["test_fraction"]
    train_idx, temp_idx = train_test_split(
        np.arange(len(lb)), test_size=test_val_ratio, stratify=lb, random_state=cfg_data["seed"]
    )
    
    # Stratified Split (Val / Test)
    val_ratio = cfg_data["val_fraction"] / test_val_ratio
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio), stratify=lb[temp_idx], random_state=cfg_data["seed"]
    )

    train_dataset = TESSDataset(gv[train_idx], lv[train_idx], mt[train_idx], lb[train_idx], fusion_type)
    val_dataset = TESSDataset(gv[val_idx], lv[val_idx], mt[val_idx], lb[val_idx], fusion_type)
    test_dataset = TESSDataset(gv[test_idx], lv[test_idx], mt[test_idx], lb[test_idx], fusion_type)

    # SMOTE-style class balancing using WeightedRandomSampler
    class_counts = np.bincount(lb[train_idx].astype(int))
    weights = 1.0 / class_counts
    sample_weights = weights[lb[train_idx].astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Workers set to 2 for Colab stability
    train_loader = DataLoader(train_dataset, batch_size=cfg_train["batch_size"], sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_train["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg_train["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

# ── Training Loops ────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for gv, lv, mt, labels in dataloader:
        gv, lv, mt, labels = gv.to(device), lv.to(device), mt.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(gv, lv, mt)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return total_loss / len(dataloader), metrics

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for gv, lv, mt, labels in dataloader:
        gv, lv, mt, labels = gv.to(device), lv.to(device), mt.to(device), labels.to(device)
        outputs = model(gv, lv, mt)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return total_loss / len(dataloader), metrics

# ── Main ──────────────────────────────────────────────────────────────────────
def main(config: dict):
    set_seed(config["data"]["seed"])
    device = get_device()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Ablation Study Fusion Type: {config['model']['fusion_type'].upper()}")
    
    train_loader, val_loader, test_loader = make_dataloaders(config)
    
    # Initialize the DualViewTransformer from model.py
    model = DualViewTransformer(config).to(device)
    logger.info(f"Total Model Parameters: {count_parameters(model):,}")
    
    use_wandb = config["logging"].get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"].get("wandb_entity", None),
            config=config
        )
        
    cfg_t = config["training"]
    
    # Optimizer & Loss Setup
    criterion = FocalLoss(alpha=cfg_t.get("focal_alpha", 0.65), gamma=cfg_t.get("focal_gamma", 1.5))
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg_t["learning_rate"]), weight_decay=float(cfg_t["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_t["epochs"])
    
    # Dynamic Checkpoint Directory based on Fusion Type
    ckpt_dir = Path(config["paths"]["checkpoint_dir"]) / config["model"]["fusion_type"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, cfg_t["epochs"] + 1):
        start_time = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch:03d} | {epoch_time:.1f}s | "
                    f"Train Loss: {train_loss:.4f} AUC: {train_metrics['roc_auc']:.4f} | "
                    f"Val Loss: {val_loss:.4f} AUC: {val_metrics['roc_auc']:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_auc": train_metrics["roc_auc"], "val_auc": val_metrics["roc_auc"],
                "train_f1": train_metrics["f1"], "val_f1": val_metrics["f1"],
                "lr": optimizer.param_groups[0]['lr']
            })
            
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            patience_counter = 0
            # Save Checkpoint
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config
            }, ckpt_dir / "best_model.pt")
            logger.info(f"  ✓ Saved new best model (AUC: {best_auc:.4f}) to {ckpt_dir.name}")
        else:
            patience_counter += 1
            
        if patience_counter >= cfg_t.get("patience", 15):
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/content/drive/MyDrive/TESS_Project/configs/config.yaml")
    # Ablation Study Switch
    parser.add_argument("--fusion_type", type=str, default="mlp", 
                        choices=["mlp", "meta_token", "film", "astrophysics"],
                        help="Choose the fusion architecture for the ablation study.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Dynamically inject the chosen fusion type into the config object
    config["model"]["fusion_type"] = args.fusion_type
    
    # Dynamically change the Weights & Biases project name so the runs are separated!
    if config.get("logging", {}).get("use_wandb"):
        config["logging"]["wandb_project"] = f"TESS-Ablation-{args.fusion_type}"
        
    main(config)