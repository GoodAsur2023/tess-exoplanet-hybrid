import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_device, load_config
from src.model import DualViewTransformer
from src.train import make_dataloaders

def generate_attention_map():
    device = get_device()
    print(f"Extracting Attention Maps on: {device}")

    # 1. Load the winning config and model
    config = load_config("configs/config.yaml")
    config["model"]["fusion_type"] = "meta_token"
    
    ckpt_path = "checkpoints/meta_token/best_model.pt"
    model = DualViewTransformer(config).to(device)
    
    # Bypass PyTorch 2.6 weights security
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # 2. Get the test data
    _, _, test_loader = make_dataloaders(config)
    
    # Grab the first batch
    gv, lv, mt, labels = next(iter(test_loader))
    
    # Find the first actual Exoplanet in the batch (Label == 1)
    planet_indices = (labels == 1).nonzero(as_tuple=True)[0]
    if len(planet_indices) == 0:
        print("No planets in this batch, try another batch!")
        return
        
    idx = planet_indices[0]
    gv_sample = gv[idx:idx+1].to(device)
    lv_sample = lv[idx:idx+1].to(device)
    mt_sample = mt[idx:idx+1].to(device)

    # 3. Manually run the front-end of your model to get the tokens
    with torch.no_grad():
        g_feat = torch.mean(model.global_cnn(gv_sample), dim=-1)
        l_feat = torch.mean(model.local_cnn(lv_sample), dim=-1)
        
        g_embed = model.global_proj(g_feat)
        l_embed = model.local_proj(l_feat)
        meta_token = model.meta_proj(mt_sample)
        
        # This is your sequence! [Meta Physics, Global View, Local View]
        seq = torch.stack([meta_token, g_embed, l_embed], dim=1)
        
        # 4. Extract the Attention Weights from the LAST Transformer Layer
        last_layer = model.transformer.layers[-1]
        
        # We manually invoke the self_attn module to force it to return the weights
        seq_norm = last_layer.norm1(seq)
        attn_output, attn_weights = last_layer.self_attn(
            seq_norm, seq_norm, seq_norm, need_weights=True, average_attn_weights=True
        )

    # 5. Plot the 3x3 Attention Matrix
    weights_matrix = attn_weights[0].cpu().numpy()
    
    labels_list = ['Stellar Physics\n(Meta Token)', 'Macro Lightcurve\n(Global View)', 'Transit Dip\n(Local View)']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights_matrix, annot=True, cmap="magma", fmt=".3f", 
                xticklabels=labels_list, yticklabels=labels_list,
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title("Transformer Cross-View Attention Map\n(Confirmed Exoplanet)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Key (Information being looked at)", fontsize=12, labelpad=10)
    plt.ylabel("Query (Token paying attention)", fontsize=12, labelpad=10)
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('outputs', exist_ok=True)
    save_path = 'outputs/meta_token_attention.pdf'
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()
    
    print(f"Attention Map saved to {save_path}")

if __name__ == "__main__":
    generate_attention_map()