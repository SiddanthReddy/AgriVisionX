from models.hybrid_model import HybridCNNTransformer

model = HybridCNNTransformer(
    num_classes=29,
    pretrained_backbones=False
)

print(model.cnn_backbone.backbone)