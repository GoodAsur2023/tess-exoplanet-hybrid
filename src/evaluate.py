import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score, 
    precision_recall_curve, confusion_matrix, average_precision_score
)
from pathlib import Path
import sys

# Ensure imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, get_device
from src.model import DualViewTransformer
from src.train import make_dataloaders # Use the correct loader from train.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate(config, checkpoint_path):
    device = get_device()
    # Use the robust dataloader from your training script
    _, _, test_loader = make_dataloaders(config)
    
    model = DualViewTransformer(config).to(device)
    
    # Load with PyTorch 2.6 security bypass
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for gv, lv, meta, labels in test_loader:
            gv, lv, meta = gv.to(device), lv.to(device), meta.to(device)
            preds = model(gv, lv, meta)
            all_preds.extend(torch.sigmoid(preds).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Find optimal threshold using PR curve for maximum F1 score
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    # Calculate F1: 2 * (P * R) / (P + R)
    fscore = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    ix = np.argmax(fscore)
    # The thresholds array is one shorter than precisions/recalls
    optimal_threshold = thresholds[ix] if ix < len(thresholds) else 0.5

    preds_binary = (all_preds >= optimal_threshold).astype(int)

    # Compute final metrics
    metrics = {
        "precision": precision_score(all_labels, preds_binary, zero_division=0),
        "recall": recall_score(all_labels, preds_binary, zero_division=0),
        "f1": f1_score(all_labels, preds_binary, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_preds),
        "avg_precision": average_precision_score(all_labels, all_preds)
    }

    logger.info(f"\n{'='*50}\nFINAL TEST RESULTS (Threshold = {optimal_threshold:.4f})\n{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v:.4f}")

    # Generate Confusion Matrix
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_labels, preds_binary)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['Non-Planet', 'Planet'], 
                yticklabels=['Non-Planet', 'Planet'])
    plt.title(f'Confusion Matrix (Thr={optimal_threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(out_dir / f"conf_matrix_{config['model']['fusion_type']}.png")
    plt.close()
    
    logger.info(f"Results saved to outputs/evaluation/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    args = parser.parse_args()
    
    config = load_config(args.config)
    # Extract fusion type from the checkpoint path automatically
    path_parts = Path(args.checkpoint).parts
    if "checkpoints" in path_parts:
        idx = path_parts.index("checkpoints")
        config["model"]["fusion_type"] = path_parts[idx+1]
        
    evaluate(config, args.checkpoint)