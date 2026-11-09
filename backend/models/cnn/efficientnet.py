"""EfficientNet-B0 CNN backbone for feature extraction.

This module wraps a `timm` EfficientNet-B0 model, strips its classification
head, and exposes a pooled embedding vector suitable for downstream fusion
with a Vision Transformer branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 feature extractor with the classifier removed.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
        out_dim: Desired output embedding dimension. A linear projection is
            applied if this differs from the backbone's native feature size.
        freeze_backbone: If True, backbone weights are frozen (only the
            optional projection layer remains trainable).
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_dim: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # `num_classes=0` makes timm return pooled features directly,
        # removing the final classification layer for us.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        native_dim = self.backbone.num_features
        self.out_dim = out_dim
        self.projection: nn.Module
        if native_dim != out_dim:
            self.projection = nn.Linear(native_dim, out_dim)
        else:
            self.projection = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a pooled embedding from a batch of images.

        Args:
            x: Image tensor of shape ``(batch_size, 3, H, W)``.

        Returns:
            Embedding tensor of shape ``(batch_size, out_dim)``.
        """
        features = self.backbone(x)
        return self.projection(features)
