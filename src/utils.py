import yaml
import random
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader


def get_device():
    """Automatically assigns the GPU if available, else falls back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Locks in random seeds so ablation study is 100% reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics(labels, preds, threshold=0.5):
    """Calculates the scientific metrics for W&B dashboard."""
    preds_binary = (preds >= threshold).astype(int)
    
    # Safely calculate AUC (handles rare edge cases in tiny batches)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5
        
    return {
        "roc_auc": auc,
        "f1": f1_score(labels, preds_binary, zero_division=0),
        "precision": precision_score(labels, preds_binary, zero_division=0),
        "recall": recall_score(labels, preds_binary, zero_division=0)
    }

def load_config(config_path="configs/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TESSDataset(Dataset):
    def __init__(self, data_dir: str):
        p = Path(data_dir)
        self.gv = torch.tensor(np.load(p / "global_views.npy"), dtype=torch.float32).unsqueeze(1)
        self.lv = torch.tensor(np.load(p / "local_views.npy"), dtype=torch.float32).unsqueeze(1)
        self.meta = torch.tensor(np.load(p / "stellar_meta.npy"), dtype=torch.float32)
        self.labels = torch.tensor(np.load(p / "labels.npy"), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.gv[idx], self.lv[idx], self.meta[idx], self.labels[idx]

def get_dataloaders(config: dict):
    dataset = TESSDataset(config["paths"]["processed_dir"])
    total = len(dataset)
    val_size = int(total * config["data"]["val_fraction"])
    test_size = int(total * config["data"]["test_fraction"])
    train_size = total - val_size - test_size
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])
    
    return train_loader, val_loader, test_loader