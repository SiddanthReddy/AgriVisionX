"""Hybrid CNN + Vision Transformer model with cross-attention fusion.

Pipeline:
    Image -> EfficientNet-B0 (CNN) -> embedding_cnn
    Image -> Swin Transformer Tiny -> embedding_vit
    (embedding_cnn, embedding_vit) -> CrossAttentionFusion -> fused
    fused -> ClassificationHead -> probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cnn.efficientnet import EfficientNetBackbone
from .transformers.swin_transformer import SwinTransformerBackbone
from .fusion.cross_attention import CrossAttentionFusion
from .classification_head import ClassificationHead


class HybridCNNTransformer(nn.Module):
    """End-to-end Hybrid CNN + ViT classifier with cross-attention fusion.

    Note on logits vs. probabilities:
        The classification head's architecture ends in a Softmax layer (per
        spec), so ``forward`` returns class probabilities. Training with
        ``nn.CrossEntropyLoss`` requires raw logits, not probabilities, so
        the training pipeline (see ``training/losses.py``) calls
        ``logits()`` on this model rather than ``forward()`` to avoid a
        double-softmax. Use ``forward()`` for inference/reporting and
        ``logits()`` when computing the training loss.

    Args:
        num_classes: Number of target classes.
        cnn_out_dim: Embedding dimension produced by the CNN backbone.
        transformer_out_dim: Embedding dimension produced by the transformer
            backbone.
        fusion_dim: Shared embedding dimension used inside the fusion module
            and passed to the classification head.
        fusion_heads: Number of attention heads in the cross-attention
            fusion module.
        head_hidden_dim: Hidden dimension of the classification head.
        dropout: Dropout probability used across fusion and the head.
        pretrained_backbones: Whether to load ImageNet-pretrained weights
            for both backbones.
        freeze_backbones: Whether to freeze both backbones' weights.
    """

    def __init__(
        self,
        num_classes: int,
        cnn_out_dim: int = 512,
        transformer_out_dim: int = 512,
        fusion_dim: int = 512,
        fusion_heads: int = 8,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        pretrained_backbones: bool = True,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()

        self.cnn_backbone = EfficientNetBackbone(
            pretrained=pretrained_backbones,
            out_dim=cnn_out_dim,
            freeze_backbone=freeze_backbones,
        )
        self.transformer_backbone = SwinTransformerBackbone(
            pretrained=pretrained_backbones,
            out_dim=transformer_out_dim,
            freeze_backbone=freeze_backbones,
        )
        self.fusion = CrossAttentionFusion(
            cnn_dim=cnn_out_dim,
            transformer_dim=transformer_out_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=dropout,
        )
        self.classification_head = ClassificationHead(
            in_dim=fusion_dim,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbones and fusion, returning the fused embedding.

        Args:
            x: Image batch of shape ``(batch_size, 3, H, W)``.

        Returns:
            Fused embedding of shape ``(batch_size, fusion_dim)``.
        """
        cnn_features = self.cnn_backbone(x)
        transformer_features = self.transformer_backbone(x)
        return self.fusion(cnn_features, transformer_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities for a batch of images.

        Args:
            x: Image batch of shape ``(batch_size, 3, H, W)``.

        Returns:
            Class probabilities of shape ``(batch_size, num_classes)``.
        """
        fused = self.encode(x)
        return self.classification_head(fused)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits (for use with ``CrossEntropyLoss``).

        Args:
            x: Image batch of shape ``(batch_size, 3, H, W)``.

        Returns:
            Logits of shape ``(batch_size, num_classes)``.
        """
        fused = self.encode(x)
        return self.classification_head.logits(fused)
