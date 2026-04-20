import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_device
from src.model import DualViewTransformer
from src.train import make_dataloaders

def analyze_errors(checkpoint_path, threshold=0.6015):
    device = get_device()
    config = load_config("configs/config.yaml")
    config["model"]["fusion_type"] = "meta_token" # Champion model
    
    # 1. Load Model
    model = DualViewTransformer(config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    
    # 2. Get Test Data
    _, _, test_loader = make_dataloaders(config)
    
    all_gv, all_lv, all_labels, all_probs = [], [], [], []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for gv, lv, meta, labels in test_loader:
            outputs = torch.sigmoid(model(gv.to(device), lv.to(device), meta.to(device)))
            all_gv.append(gv.numpy())
            all_lv.append(lv.numpy())
            all_labels.append(labels.numpy())
            all_probs.append(outputs.cpu().numpy())

    gv = np.concatenate(all_gv, axis=0)
    lv = np.concatenate(all_lv, axis=0)
    labels = np.concatenate(all_labels, axis=0).flatten()
    probs = np.concatenate(all_probs, axis=0).flatten()
    preds = (probs >= threshold).astype(int)

    # 3. Identify Errors
    # False Positives: Pred=1, Actual=0
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    # False Negatives: Pred=0, Actual=1
    fn_idx = np.where((preds == 0) & (labels == 1))[0]

    print(f"Found {len(fp_idx)} False Positives and {len(fn_idx)} False Negatives.")

    # 4. Plot representative errors
    out_dir = Path("outputs/error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_samples(indices, title, filename):
        if len(indices) == 0: return
        num_to_plot = min(3, len(indices))
        fig, axes = plt.subplots(num_to_plot, 2, figsize=(12, 4*num_to_plot))
        for i in range(num_to_plot):
            idx = indices[i]
            # Global View
            axes[i, 0].plot(gv[idx, 0], color='gray', alpha=0.5)
            axes[i, 0].set_title(f"{title} (Global) - Prob: {probs[idx]:.4f}")
            # Local View
            axes[i, 1].plot(lv[idx, 0], color='crimson')
            axes[i, 1].set_title(f"{title} (Local View)")
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.show()

    print("Plotting samples...")
    plot_samples(fp_idx, "False Positive", "false_positives.png")
    plot_samples(fn_idx, "False Negative", "false_negatives.png")

if __name__ == "__main__":
    analyze_errors("checkpoints/meta_token/best_model.pt")