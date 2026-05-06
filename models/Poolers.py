"""
models/Poolers.py — Feature Aggregation Layers
==============================================
Provides various ways to aggregate a sequence of article embeddings
into a single daily or window-level representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooler(nn.Module):
    """Learns a global context vector to weight input embeddings."""
    def __init__(self, emb_dim: int = 1024):
        super().__init__()
        self.context_vector = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.attention = nn.Linear(emb_dim, emb_dim)
        self.output_projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, emb_dim)
            mask: Optional mask for padding tokens/news
        Returns:
            pooled: (batch, emb_dim)
        """
        # Calculate scores: (batch, seq_len, 1)
        # Using a simple additive/dot attention against the learned context
        weights = torch.matmul(F.tanh(self.attention(x)), self.context_vector.transpose(1, 2))
        
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum: (batch, emb_dim)
        pooled = torch.sum(x * weights, dim=1)
        return self.output_projection(pooled)


class IdentityPooler(nn.Module):
    """Simple Max/Avg aggregation (no learnable params)."""
    def __init__(self, mode="max"):
        super().__init__()
        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == "max":
            # Masked max
            if mask is not None:
                x = x.masked_fill(mask == 0, -1e9)
            return torch.max(x, dim=1)[0]
        else:
            # Masked mean
            if mask is not None:
                count = mask.sum(dim=1)
                return torch.sum(x * mask, dim=1) / (count + 1e-9)
            return torch.mean(x, dim=1)
