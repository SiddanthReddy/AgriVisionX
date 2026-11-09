"""Classification head applied to the fused embedding.

Architecture: Linear -> GELU -> Dropout -> Linear -> Softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Feed-forward classification head with a softmax output.

    Args:
        in_dim: Dimensionality of the input (fused) embedding.
        hidden_dim: Dimensionality of the hidden layer.
        num_classes: Number of output classes.
        dropout: Dropout probability between the two linear layers.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities from a fused embedding.

        Note:
            This returns softmax probabilities, not raw logits. If pairing
            with ``nn.CrossEntropyLoss`` (which expects logits), use the
            ``logits`` method instead of ``forward`` when computing the
            training loss, to avoid a double-softmax.

        Args:
            x: Fused embedding of shape ``(batch_size, in_dim)``.

        Returns:
            Class probabilities of shape ``(batch_size, num_classes)``.
        """
        return self.softmax(self.logits(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw (pre-softmax) class logits.

        Args:
            x: Fused embedding of shape ``(batch_size, in_dim)``.

        Returns:
            Logits of shape ``(batch_size, num_classes)``.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)
