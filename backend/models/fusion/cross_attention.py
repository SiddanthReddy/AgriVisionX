"""Cross-Attention Fusion module.

Fuses a CNN embedding and a Vision Transformer embedding via bidirectional
multi-head cross-attention: CNN features attend to transformer features and
transformer features attend to CNN features. The attended representations
are concatenated, projected, combined with a residual connection, and
normalized.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion of two embedding streams.

    Each input embedding is treated as a single-token sequence so that
    standard multi-head attention can be applied between the two modalities.

    Args:
        cnn_dim: Dimensionality of the CNN embedding.
        transformer_dim: Dimensionality of the transformer embedding.
        fusion_dim: Dimensionality of the fused output embedding.
        num_heads: Number of attention heads.
        dropout: Dropout probability applied inside attention and the
            fusion projection.
    """

    def __init__(
        self,
        cnn_dim: int = 512,
        transformer_dim: int = 512,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        # Project both modalities into a shared dimension before attention.
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.transformer_proj = nn.Linear(transformer_dim, fusion_dim)

        # CNN queries attend to transformer keys/values.
        self.cnn_to_transformer_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Transformer queries attend to CNN keys/values.
        self.transformer_to_cnn_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Combine both attended streams into a single fused embedding.
        self.fusion_projection = nn.Linear(fusion_dim * 2, fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self, cnn_features: torch.Tensor, transformer_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse CNN and transformer embeddings via cross-attention.

        Args:
            cnn_features: Tensor of shape ``(batch_size, cnn_dim)``.
            transformer_features: Tensor of shape
                ``(batch_size, transformer_dim)``.

        Returns:
            Fused embedding of shape ``(batch_size, fusion_dim)``.
        """
        # Project to shared dimension and add a singleton sequence axis so
        # nn.MultiheadAttention can operate on them as length-1 sequences.
        cnn_tok = self.cnn_proj(cnn_features).unsqueeze(1)
        trans_tok = self.transformer_proj(transformer_features).unsqueeze(1)

        # CNN attends to transformer.
        cnn_attended, _ = self.cnn_to_transformer_attn(
            query=cnn_tok, key=trans_tok, value=trans_tok
        )
        # Transformer attends to CNN.
        trans_attended, _ = self.transformer_to_cnn_attn(
            query=trans_tok, key=cnn_tok, value=cnn_tok
        )

        cnn_attended = cnn_attended.squeeze(1)
        trans_attended = trans_attended.squeeze(1)

        # Residual connections back to the projected (pre-attention) tokens.
        cnn_residual = cnn_attended + cnn_tok.squeeze(1)
        trans_residual = trans_attended + trans_tok.squeeze(1)

        fused = torch.cat([cnn_residual, trans_residual], dim=-1)
        fused = self.fusion_projection(fused)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused
