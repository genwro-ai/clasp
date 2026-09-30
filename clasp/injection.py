"""Application-only LoRA wrappers for the frozen UNet's cross-attention projections.

`CachedLoRALinear` owns no weights. It reads the factors (A, B) that the hypernetwork cached for
its layer and returns `W0 x + (x A) B`. On the projections that read the prompt (to_k, to_v) the
update is multiplied by the token mask, so it acts only at the concept's class-word positions.
"""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn

DEFAULT_TARGETS: tuple[str, ...] = ("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0")


def _match(full_name: str, target: str) -> bool:
    """Suffix match ("attn2.to_k"); "prefix*suffix" additionally requires the prefix."""
    if "*" in target:
        prefix, suffix = target.split("*", 1)
        return full_name.startswith(prefix) and full_name.endswith(suffix)
    return full_name.endswith(target)


class CachedLoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, layer_name: str = ""):
        super().__init__()
        self.original = original_linear
        self.layer_name = layer_name
        self._parent_ref = None  # weakref to the module holding `.hyper` (the UNet)

    def set_parent(self, parent: nn.Module) -> None:
        self._parent_ref = weakref.ref(parent)

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        parent = self._parent_ref() if self._parent_ref is not None else None
        hyper = getattr(parent, "hyper", None) if parent is not None else None
        if hyper is None or not hyper.lora_enabled:
            return out
        cached = hyper.get_cached_lora(self.layer_name)
        if cached is None:
            return out
        x_L, x_R = cached  # x_L: [Bc, in, rank], x_R: [Bc, rank, out]
        if x_L.shape[0] == 1 and x.shape[0] > 1:
            x_L = x_L.expand(x.shape[0], -1, -1)
            x_R = x_R.expand(x.shape[0], -1, -1)
        lora = (x.float() @ x_L.float()) @ x_R.float()
        mask = hyper.get_token_mask()
        if mask is not None and x.dim() == 3 and mask.shape[-1] == x.shape[1]:
            m = mask.to(device=lora.device, dtype=lora.dtype)
            if m.shape[0] == 1 and x.shape[0] > 1:
                m = m.expand(x.shape[0], -1)
            lora = lora * m[:, :, None]   # update only at the concept's token positions
        return out + lora.to(out.dtype)


def inject_lora(module: nn.Module, target_modules: tuple[str, ...] = DEFAULT_TARGETS,
                name: str = "") -> list[tuple[str, CachedLoRALinear]]:
    """Recursively replace matching nn.Linear children by CachedLoRALinear wrappers.
    Returns [(dotted_name, wrapper)]; the dotted name is the hypernetwork's layer key."""
    wrapped: list[tuple[str, CachedLoRALinear]] = []
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Linear) and any(_match(full_name, t) for t in target_modules):
            wrapper = CachedLoRALinear(child, layer_name=full_name).to(next(child.parameters()).device)
            setattr(module, child_name, wrapper)
            wrapped.append((full_name, wrapper))
        else:
            wrapped.extend(inject_lora(child, target_modules, full_name))
    return wrapped
