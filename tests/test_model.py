import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config
from src.model import DualViewTransformer

def test_model_dimensions():
    print("Loading config...")
    config = load_config("configs/config.yaml")
    
    print(f"Testing Architecture: {config['model']['fusion_type']}")
    model = DualViewTransformer(config)
    
    # Create fake batch data (Batch Size = 4)
    dummy_gv = torch.randn(4, 1, 2048)
    dummy_lv = torch.randn(4, 1, 201)
    dummy_meta = torch.randn(4, 3) # Temp, Gravity, Radius
    
    print("Running forward pass...")
    output = model(dummy_gv, dummy_lv, dummy_meta)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1), "Output dimension is incorrect! Should be (Batch, 1)"
    print("All dimension tests passed successfully!")

if __name__ == "__main__":
    test_model_dimensions()