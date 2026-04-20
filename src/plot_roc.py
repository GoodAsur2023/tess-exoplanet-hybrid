import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# Ensure imports work from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_device, load_config
from src.model import DualViewTransformer
from src.train import make_dataloaders

def plot_ablation_roc():
    device = get_device()
    print(f"Evaluating on device: {device}")

    fusion_types = ['mlp', 'meta_token', 'film', 'astrophysics']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Late Fusion (MLP)', 'Meta Token (Native)', 'FiLM (Modulation)', 'Science Way (Astrophysics)']

    plt.figure(figsize=(10, 8))

    for f_type, color, label in zip(fusion_types, colors, labels):
        ckpt_path = f"checkpoints/{f_type}/best_model.pt"

        if not os.path.exists(ckpt_path):
            print(f" Missing checkpoint for {f_type}. Skipping...")
            continue

        print(f"Loading and evaluating {f_type}...")

        # 1. Load config and force the fusion type
        config = load_config("configs/config.yaml")
        config["model"]["fusion_type"] = f_type

        # 2. Build model and load winning weights
        model = DualViewTransformer(config).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.eval()

        # 3. Load the Test Data
        _, _, test_loader = make_dataloaders(config)

        all_preds, all_labels = [], []

        # 4. Run Inference
        with torch.no_grad():
            for gv, lv, mt, batch_labels in test_loader:
                gv, lv, mt = gv.to(device), lv.to(device), mt.to(device)
                outputs = model(gv, lv, mt)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_preds.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())

        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{label} (AUC = {roc_auc:.4f})')

    # Format the plot
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Ablation Study: Test Set ROC Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    # Save the plot
    os.makedirs('outputs', exist_ok=True)
    save_path = 'outputs/ablation_roc_comparison.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    print(f"\n Evaluation complete! Plot saved to {save_path}")

if __name__ == "__main__":
    plot_ablation_roc()