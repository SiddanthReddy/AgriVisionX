import sys
from pathlib import Path
import torch

# Add backend folder to path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_model import HybridCNNTransformer

print("Creating model...")

model = HybridCNNTransformer(
    num_classes=29,
    pretrained_backbones=False
)

checkpoint_path = Path(__file__).parent.parent / "model" / "best_model.pth"

print("Loading checkpoint:", checkpoint_path)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("Checkpoint keys:", checkpoint.keys())

model.load_state_dict(checkpoint["model_state_dict"])

print("✅ Model loaded successfully")

# Test forward pass
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(x)

print("Output shape:", output.shape)
print("✅ Forward pass successful")