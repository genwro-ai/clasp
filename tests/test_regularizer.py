"""The trace form of the regularizer equals the explicit Frobenius distance (CPU)."""
import torch

from clasp.regularizer import reg_dw


def explicit(now, targets):
    terms = []
    for n in targets:
        a1, b1 = now[n]
        a2, b2 = targets[n]
        d = a1 @ b1 - a2 @ b2                                   # [k, in, out]
        terms.append((d.pow(2).sum((1, 2)) / (a1.shape[1] * b1.shape[2])).mean())
    return torch.stack(terms).mean()


def test_trace_equals_frobenius():
    g = torch.Generator().manual_seed(0)
    shapes = {"l0": (320, 768), "l1": (1280, 1280), "l2": (640, 64)}
    k, r = 5, 4
    now = {n: (torch.randn(k, i, r, generator=g, dtype=torch.float64),
               torch.randn(k, r, o, generator=g, dtype=torch.float64)) for n, (i, o) in shapes.items()}
    tgt = {n: (a + 0.1 * torch.randn(a.shape, generator=g, dtype=torch.float64),
               b + 0.1 * torch.randn(b.shape, generator=g, dtype=torch.float64))
           for n, (a, b) in now.items()}
    got, want = reg_dw(now, tgt), explicit(now, tgt)
    assert torch.allclose(got, want, rtol=1e-10, atol=0), (float(got), float(want))
    assert float(reg_dw(now, now)) < 1e-9 * float(want)


if __name__ == "__main__":
    test_trace_equals_frobenius()
    print("OK")
