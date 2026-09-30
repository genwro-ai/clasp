"""Localization module: box encoding, the fixed box mask inside(p, b), and the gated
attention branch (GSA) that reads the placement tokens inside every cross-attention layer.

The learnable parts of the branch (token head, FiLM head, per-layer q/k/v/o projections and
per-layer gates) are attributes of the hypernetwork (`hypernet.Hypernetwork`), so a
checkpoint holds them together with the content heads.
"""

from __future__ import annotations

import torch

EDGE_SHARPNESS = 40.0   # lambda of the soft box indicator inside(p, b); about a tenth of the frame per edge


def fourier_box(box) -> torch.Tensor:
    """(cx, cy, w, h) in [0, 1] -> [1, 64]: sine and cosine of each coordinate at 8 frequencies."""
    v = torch.as_tensor(box, dtype=torch.float32)
    k = 2.0 ** torch.arange(8) * torch.pi
    ang = v[:, None] * k[None, :]                       # [4, 8]
    return torch.cat([ang.sin(), ang.cos()], dim=1).reshape(1, 64)


def box_mask(grid: torch.Tensor, cx, cy, bw, bh, sharpness: float = EDGE_SHARPNESS) -> torch.Tensor:
    """inside(p, b) for cell centres grid [n, 2] in [0, 1]^2: product of one sigmoid per edge."""
    sh = float(sharpness)
    return (torch.sigmoid(sh * (grid[:, 0] - (cx - bw / 2)))
            * torch.sigmoid(sh * ((cx + bw / 2) - grid[:, 0]))
            * torch.sigmoid(sh * (grid[:, 1] - (cy - bh / 2)))
            * torch.sigmoid(sh * ((cy + bh / 2) - grid[:, 1])))


class GroundedAttnProcessor:
    """Cross-attention unchanged, plus a parallel gated branch:

        out_p += kappa * tanh(g_l) * inside(p, b) * r_p,

    added before the output projection. `kappa` is `manager.ground_gain` (1.0 when unset, as
    during training). The branch runs only in the conditional pass (LoRA enabled).
    """

    def __init__(self, attn2_name: str, manager):
        self.name = attn2_name
        self.manager = manager

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kw):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        ndim = hidden_states.ndim
        if ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            ctx = attn.norm_encoder_hidden_states(ctx)
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(ctx))
        v = attn.head_to_batch_dim(attn.to_v(ctx))
        # attention logits in fp32, as the default SDPA processor keeps them
        s = torch.baddbmm(torch.zeros(q.shape[0], q.shape[1], k.shape[1],
                                      device=q.device, dtype=torch.float32),
                          q.float(), k.float().transpose(-1, -2), beta=0, alpha=attn.scale)
        out = torch.bmm(s.softmax(dim=-1).to(v.dtype), v)

        if getattr(self.manager, "ground_gsa", False) and encoder_hidden_states is not None:
            g = self.manager.get_ground(self.name) \
                if getattr(self.manager, "lora_enabled", True) else None
            if g is not None:
                _, gate = g
                read = self.manager.gsa_read(self.name, hidden_states)
                if read is not None:
                    n_img = hidden_states.shape[1]
                    if ndim == 4:
                        gh, gw = h, w
                    else:
                        gh = gw = int(n_img ** 0.5)
                    inside = self.manager.geo_inside(gh, gw, out.device, out.dtype)  # [n,1]
                    heads = attn.heads
                    read_h = attn.head_to_batch_dim(read)                            # [B*hd, n, d/hd]
                    # head_to_batch_dim is batch-major (index b*heads+h), so a per-sample mask
                    # is expanded with repeat_interleave
                    ins = inside.unsqueeze(0) if inside.ndim == 2 \
                        else inside.repeat_interleave(heads, dim=0)                 # [B*hd, n, 1]
                    gain = float(getattr(self.manager, "ground_gain", 1.0))
                    out = out + gain * torch.tanh(gate).to(out.dtype) * ins * read_h
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[1](attn.to_out[0](out))
        if ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def set_grounded(unet, manager, enable: bool = True) -> int:
    """Install the grounded processor on every attn2; enable=False restores the default."""
    from diffusers.models.attention_processor import AttnProcessor
    n = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and hasattr(mod, "set_processor"):
            mod.set_processor(GroundedAttnProcessor(name, manager) if enable else AttnProcessor())
            n += 1
    return n
