"""Swin Transformer Tiny backbone for feature extraction.

Wraps the `timm` implementation of Swin-T (``swin_tiny_patch4_window7_224``),
strips the classification head, and exposes a pooled embedding vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class SwinTransformerBackbone(nn.Module):
    """Swin Transformer Tiny feature extractor with the classifier removed.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
        out_dim: Desired output embedding dimension. A linear projection is
            applied if this differs from the backbone's native feature size.
        img_size: Input image resolution expected by the model. Swin-T's
            default pretrained configuration expects 224x224 inputs.
        freeze_backbone: If True, backbone weights are frozen (only the
            optional projection layer remains trainable).
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_dim: int = 512,
        img_size: int = 224,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
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
