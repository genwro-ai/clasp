"""Output-space regularizer, evaluated without forming any update.

For two rank-r products A1 B1 and A2 B2,
    ||A1 B1 - A2 B2||_F^2 = tr(G11 H11) - 2 tr(G12 H12) + tr(G22 H22),
with G_ij = A_i^T A_j and H_ij = B_j B_i^T, all r x r. Each layer's term is divided by
d_in * d_out, so the penalty is a mean over weight entries, then averaged over layers and
earlier concepts.
"""

from __future__ import annotations

import torch


def reg_dw(now, targets):
    """now, targets: {layer: (A [k, in, r], B [k, r, out])} for the k earlier concepts."""
    terms = []
    for n in targets:
        a1, b1 = now[n]
        a2, b2 = targets[n]
        g1 = a1.transpose(1, 2) @ a1              # [k, r, r]
        g2 = a2.transpose(1, 2) @ a2
        gx = a1.transpose(1, 2) @ a2
        h1 = b1 @ b1.transpose(1, 2)
        h2 = b2 @ b2.transpose(1, 2)
        hx = b2 @ b1.transpose(1, 2)
        t11 = (g1 * h1.transpose(1, 2)).sum((1, 2))
        t22 = (g2 * h2.transpose(1, 2)).sum((1, 2))
        t12 = (gx * hx.transpose(1, 2)).sum((1, 2))
        terms.append(((t11 - 2 * t12 + t22) / (a1.shape[1] * b1.shape[2])).mean())
    return torch.stack(terms).mean()
