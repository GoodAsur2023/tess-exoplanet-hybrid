"""
evaluate.py — Post-training evaluation and explainability visualisation.

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model
from src.train import make_dataloaders
from src.utils import (
    compute_metrics,
    find_best_threshold,
    get_device,
    load_checkpoint,
    load_config,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_labels, sample_embeddings = [], [], []
    for i, (gv, lv, labels) in enumerate(loader):
        gv, lv = gv.to(device), lv.to(device)
        logits, tokens = model(gv, lv)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        if i == 0 and tokens is not None:
            sample_embeddings = [
                (gv[j].cpu(), lv[j].cpu(), tokens[j].cpu(), labels[j].item(), probs[j])
                for j in range(min(4, len(labels)))
            ]
    return np.concatenate(all_probs), np.concatenate(all_labels), sample_embeddings


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_roc_and_pr(y_true, y_prob, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}", color="#4C72B0")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend()
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    axes[1].plot(rec, prec, lw=2, label=f"AP = {ap:.3f}", color="#DD8452")
    axes[1].axhline(baseline, color="k", lw=0.8, linestyle="--", label=f"Baseline = {baseline:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()
    path = out_dir / "roc_pr_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC/PR plot → {path}")


def plot_confusion_matrix(y_true, y_prob, threshold, out_dir):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["False Positive", "Planet"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix → {path}")


def plot_attention_maps(sample_embeddings, global_length, local_length, out_dir):
    if not sample_embeddings:
        return
    fig, axes = plt.subplots(len(sample_embeddings), 2, figsize=(13, 3 * len(sample_embeddings)))
    if len(sample_embeddings) == 1:
        axes = [axes]
    for row_idx, (gv, lv, tokens, label, prob) in enumerate(sample_embeddings):
        importance = tokens.norm(dim=-1).numpy()
        n_global_tokens = len(importance) * global_length // (global_length + local_length)
        global_importance = importance[:n_global_tokens]
        local_importance  = importance[n_global_tokens:]
        gv_np = gv.numpy()
        lv_np = lv.numpy()
        global_imp_up = np.interp(np.linspace(0, 1, global_length),
                                  np.linspace(0, 1, len(global_importance)), global_importance)
        local_imp_up  = np.interp(np.linspace(0, 1, local_length),
                                  np.linspace(0, 1, len(local_importance)),  local_importance)
        title = f"{'Planet' if label == 1 else 'FP'} p={prob:.3f}"
        phase_global = np.linspace(-0.5, 0.5, global_length)
        phase_local  = np.linspace(-0.075, 0.075, local_length)
        ax = axes[row_idx][0]
        ax.plot(phase_global, gv_np, color="#4C72B0", lw=0.7)
        ax.twinx().fill_between(phase_global, global_imp_up, alpha=0.25, color="#DD8452")
        ax.set(xlabel="Phase", ylabel="Normalised flux", title=f"Global view — {title}")
        ax = axes[row_idx][1]
        ax.plot(phase_local, lv_np, color="#4C72B0", lw=0.7)
        ax.twinx().fill_between(phase_local, local_imp_up, alpha=0.25, color="#DD8452")
        ax.set(xlabel="Phase", ylabel="Normalised flux", title=f"Local view — {title}")
    plt.tight_layout()
    path = out_dir / "attention_maps.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved attention maps → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate_checkpoint(config: dict, checkpoint_path: str) -> None:
    set_seed(config["data"]["seed"])
    device = get_device()
    state = load_checkpoint(checkpoint_path, device)
    model = build_model(config, device)
    model.load_state_dict(state["model_state"])
    logger.info(f"Loaded model from epoch {state['epoch']}.")
    _, _, test_loader = make_dataloaders(config)
    y_prob, y_true, sample_embeddings = run_inference(model, test_loader, device)
    best_thresh = find_best_threshold(y_true, y_prob, metric="f1")
    metrics = compute_metrics(y_true, y_prob, threshold=best_thresh)
    logger.info(f"\n{'='*55}\nTEST RESULTS (threshold = {best_thresh:.2f})\n{'='*55}")
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v:.4f}")
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_and_pr(y_true, y_prob, out_dir)
    plot_confusion_matrix(y_true, y_prob, best_thresh, out_dir)
    plot_attention_maps(sample_embeddings, config["data"]["global_view_length"],
                        config["data"]["local_view_length"], out_dir)
    logger.info(f"\nAll evaluation plots saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_checkpoint(config, args.checkpoint)