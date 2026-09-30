"""Per-layer content head: task embedding -> rank-r factors (A, B) of one adapted projection.

The right branch (which produces B) has a zero-initialized output layer, so at construction the
generated update is exactly zero. The scalar gain alpha is folded into A.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HyperHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, rank: int = 4, hidden: int = 50,
                 alpha_init: float = 1.0, learn_alpha: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.left = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, in_dim * rank))
        self.right = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, out_dim * rank))
        nn.init.zeros_(self.right[-1].weight)
        nn.init.zeros_(self.right[-1].bias)
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """c: [B, cond_dim] -> (alpha[B], A[B, in, r], B[B, r, out]) with alpha folded into A."""
        b = c.shape[0]
        x_L = self.left(c).view(b, self.in_dim, self.rank)
        x_R = self.right(c).view(b, self.rank, self.out_dim)
        alpha = self.alpha.to(c.dtype)
        x_L = alpha * x_L
        return alpha.expand(b).clone(), x_L, x_R
