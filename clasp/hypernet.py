"""The hypernetwork: content heads (one per adapted layer) plus the localization module.

Concept t is represented by a task embedding v_t, a random vector drawn from seed 1234 + t and
orthogonalized by Gram--Schmidt against the embeddings of earlier concepts. The content heads
map v_t to the rank-r factors of every adapted cross-attention projection. The generated update
does not depend on the timestep, so it is computed once per generation and cached.

The localization module maps (v_t, box) to M placement tokens (token head), v_t to a FiLM
modulation (modulation head), and reads the tokens inside each cross-attention layer through
narrow per-layer projections behind a zero-initialized gate (see `localization.py`).

"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from .hyper_head import HyperHead
from .injection import DEFAULT_TARGETS, _match, inject_lora
from .localization import box_mask, fourier_box

LoraPair = tuple[torch.Tensor, torch.Tensor]
KEY_SEED = 1234


def _key(name: str) -> str:
    return name.replace(".", "__")


class Hypernetwork(nn.Module):
    def __init__(self, n_tasks: int, cond_dim: int, task_cond: dict, clip_dim: int = 768):
        super().__init__()
        self.heads = nn.ModuleDict()           # sanitized layer name -> HyperHead
        self.layer_names: list[str] = []        # dotted names, cache-key order
        self._cache: dict[str, LoraPair] = {}
        self._token_mask: torch.Tensor | None = None
        self.lora_scale = 1.0                   # inference-time scale s_lora
        self.lora_scale_map = None              # optional [(pattern, scale)], first match wins
        self._ctx: dict[str, object] = {"pooled": None, "task_idx": None, "token_mask": None}
        self.lora_enabled = True

        tc = task_cond or {}
        self.task_cond_enabled = bool(tc.get("enabled", False)) and n_tasks > 0
        self.ground_cond = bool(tc.get("ground_cond", False))
        self.ground_gsa = bool(tc.get("ground_gsa", False)) and self.ground_cond
        self.ground_gsa_tokens = int(tc.get("ground_gsa_tokens", 4))   # M
        self.ground_tok_dim = int(clip_dim)     # width d of the placement tokens
        self.bs_dilate = 3                      # bootstrap box dilation, in latent cells
        self.cond_box = None                    # (cx, cy, w, h) in [0, 1], or a list per sample
        self.ground_head = None                 # token head h^tok
        self.ground_gates = None                # one scalar g_l per layer
        self.ground_gsa_mods = None             # per-layer q/k/v/o projections of width 64
        self.ground_film = None                 # modulation head h^FiLM
        self._ground_vec = None
        self._ground_film_gb = None
        self._ground_box = None
        self._geo_grid_cache = {}
        if self.ground_cond:
            gin = int(tc.get("key_dim") or clip_dim) + 64
            _m = self.ground_gsa_tokens if self.ground_gsa else 1
            self.ground_head = nn.Sequential(nn.Linear(gin, 256), nn.SiLU(),
                                             nn.Linear(256, clip_dim * _m))
            # only the gates start at zero; a zero token head as well would leave both
            # the gate and the head without gradient at initialization
            with torch.no_grad():
                self.ground_head[-1].weight.mul_(0.1)
                nn.init.zeros_(self.ground_head[-1].bias)
        if self.ground_gsa:
            self.ground_film = nn.Linear(int(tc.get("key_dim") or clip_dim), 128)
            nn.init.zeros_(self.ground_film.weight)
            nn.init.zeros_(self.ground_film.bias)
        if self.task_cond_enabled:
            self.register_buffer("ortho_basis", torch.zeros(n_tasks, cond_dim))   # unit rows z_i
            self.register_buffer("basis_count", torch.zeros((), dtype=torch.long))
            self.register_buffer("canon_pooled", torch.zeros(n_tasks, cond_dim))  # keys v~_t

    # ------------------------------------------------------------------ setup
    def add_head(self, layer_name: str, head: HyperHead) -> None:
        self.heads[_key(layer_name)] = head
        self.layer_names.append(layer_name)

    def init_ground_gates(self, layer_names) -> None:
        if self.ground_head is None or self.ground_gates is not None:
            return
        self.ground_gates = nn.ParameterDict(
            {_key(n): nn.Parameter(torch.zeros(1)) for n in layer_names if n.endswith("to_q")})

    def init_ground_gsa(self, layer_dims: dict) -> None:
        """layer_dims: to_q name -> in_features. Narrow read projections per layer."""
        if not self.ground_gsa or self.ground_gsa_mods is not None:
            return
        mods = {}
        for n, d in layer_dims.items():
            mods[_key(n)] = nn.ModuleDict({
                "q": nn.Linear(d, 64, bias=False),
                "k": nn.Linear(self.ground_tok_dim, 64, bias=False),
                "v": nn.Linear(self.ground_tok_dim, 64, bias=False),
                "o": nn.Linear(64, d, bias=False),
            })
        self.ground_gsa_mods = nn.ModuleDict(mods)

    # ------------------------------------------------------------------ task embeddings
    @torch.no_grad()
    def set_canonical(self, task_idx: int) -> None:
        """Draw the key of task t from seed 1234 + t (start of the task)."""
        if not self.task_cond_enabled:
            return
        g = torch.Generator().manual_seed(KEY_SEED + int(task_idx))
        d = self.canon_pooled.shape[1]
        key = torch.zeros(d)
        key[:d] = torch.randn(d, generator=g)
        self.canon_pooled[task_idx] = key.to(self.canon_pooled.dtype)

    def condition(self, pooled: torch.Tensor, task_idx: int | None = None) -> torch.Tensor:
        """-> [1, d_v] conditioning of task t: its key, Gram--Schmidt-projected against the
        frozen directions of earlier tasks. `pooled` only sets the dtype."""
        cond = pooled.to(next(self.parameters()).dtype)
        if task_idx is None or not self.task_cond_enabled:
            return cond
        canon = self.canon_pooled[task_idx]
        if not torch.any(canon != 0):
            raise RuntimeError(f"key of task {task_idx} not set "
                               "(set_canonical at task start / load a checkpoint that carries it)")
        h = canon.to(cond.dtype).unsqueeze(0)                              # [1, D]
        n_prev = min(int(self.basis_count.item()), int(task_idx))
        if n_prev > 0:
            basis = self.ortho_basis[:n_prev].to(h.dtype)                 # [n, D]
            h = h - (h @ basis.t()) @ basis
        return h

    @torch.no_grad()
    def freeze_task_basis(self, task_idx: int) -> None:
        """After task t: freeze its direction z_t into the orthogonal basis."""
        if not self.task_cond_enabled:
            return
        h = self.condition(self.canon_pooled[task_idx:task_idx + 1], task_idx)[0]
        z = h / h.norm().clamp_min(1e-8)
        self.ortho_basis[task_idx] = z
        self.basis_count.fill_(max(int(self.basis_count.item()), task_idx + 1))

    # ------------------------------------------------------------------ localization
    def set_ground(self, task_idx: int | None, box=None) -> None:
        """Placement tokens and FiLM for (concept, box). `box` is one (cx, cy, w, h), a list
        with one box per sample, or None (full frame). task_idx=None disables the branch."""
        if self.ground_head is None or task_idx is None:
            self._ground_vec = None
            self._ground_box = None
            return
        multi = box is not None and not isinstance(box[0], (int, float))
        boxes = [tuple(b) for b in box] if multi else \
            [tuple(box) if box is not None else (0.5, 0.5, 1.0, 1.0)]
        # v_t, the key after Gram--Schmidt, as for the content heads
        key = self.condition(self.canon_pooled[task_idx:task_idx + 1], task_idx).float()
        dev = key.device
        keys = key.expand(len(boxes), -1)
        fb = torch.cat([fourier_box(b) for b in boxes], dim=0)
        gv = self.ground_head(torch.cat([keys, fb.to(dev)], dim=-1))
        self._ground_vec = gv
        if self.ground_gsa:
            self._ground_vec = gv.reshape(len(boxes), self.ground_gsa_tokens, -1)  # [B, M, D]
            self._ground_film_gb = self.ground_film(keys)                   # [B, 128] (gamma|beta)
            self._ground_box = boxes if multi else boxes[0]

    def geo_inside(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """[n, 1] (or [B, n, 1] for a box per sample) mask inside(p, b) on an h x w map."""
        key = ("xy", h, w)
        grid = self._geo_grid_cache.get(key)
        if grid is None:
            ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
            xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            self._geo_grid_cache[key] = grid
        g = grid.to(device=device, dtype=torch.float32)
        gb = self._ground_box or (0.5, 0.5, 1.0, 1.0)
        if isinstance(gb, list):
            return torch.stack([box_mask(g, *b) for b in gb], 0).to(dtype).unsqueeze(-1)
        return box_mask(g, *gb).to(dtype).unsqueeze(1)

    def gsa_read(self, attn2_name: str, x: torch.Tensor) -> torch.Tensor | None:
        """Read of the placement branch: x [B, n, d] -> r [B, n, d], before gate and mask."""
        if not self.ground_gsa or self._ground_vec is None:
            return None
        k = _key(attn2_name + ".to_q")
        if k not in self.ground_gsa_mods:
            return None
        m = self.ground_gsa_mods[k]
        e = self._ground_vec.to(x.device)                              # [B, M, d]
        gb = self._ground_film_gb.to(x.device)                         # [B, 128]
        gamma, beta = gb[:, :64], gb[:, 64:]
        xf = x.float()
        q = m["q"](xf)                                                 # [B, n, 64]
        kk = m["k"](e.float())                                         # [B, M, 64]
        v = m["v"](e.float()) * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        att = torch.softmax(q @ kk.transpose(-1, -2) / 8.0, dim=-1)    # [B, n, M]
        return m["o"](att @ v).to(x.dtype)                             # [B, n, d]

    def get_ground(self, attn2_name: str):
        """(tokens, gate) for this attn2 module, or None."""
        if self._ground_vec is None or self.ground_gates is None:
            return None
        k = _key(attn2_name + ".to_q")
        if k not in self.ground_gates:
            return None
        return self._ground_vec, self.ground_gates[k]

    # ------------------------------------------------------------------ LoRA toggle
    def enable_lora(self) -> None:
        self.lora_enabled = True

    @contextmanager
    def no_lora(self):
        prev = self.lora_enabled
        self.lora_enabled = False
        try:
            yield
        finally:
            self.lora_enabled = prev

    # ------------------------------------------------------------------ generated update
    def set_context(self, clip_pooled: torch.Tensor, task_idx: int | None = None,
                    token_mask: torch.Tensor | None = None) -> None:
        self._ctx = {"pooled": clip_pooled, "task_idx": task_idx, "token_mask": token_mask}

    def compute_and_cache_loras(self) -> None:
        """Run all content heads once and cache (A, B) per layer. Timestep-independent."""
        clip_pooled = self._ctx["pooled"]
        task_idx = self._ctx.get("task_idx")
        token_mask = self._ctx.get("token_mask")
        if clip_pooled is None:
            raise RuntimeError("compute_and_cache_loras called without context")
        cond = self.condition(clip_pooled, task_idx)
        cache: dict[str, LoraPair] = {}
        for name in self.layer_names:
            _, x_L, x_R = self.heads[_key(name)](cond)
            sc = self._scale_for(name)
            if sc != 1.0:
                x_L = x_L * sc
            cache[name] = (x_L, x_R)
        self._cache = cache
        self._token_mask = token_mask

    def get_token_mask(self) -> torch.Tensor | None:
        return self._token_mask

    def _scale_for(self, layer_name: str) -> float:
        if self.lora_scale_map:
            for pattern, sc in self.lora_scale_map:
                if _match(layer_name, pattern):
                    return float(sc)
        return self.lora_scale

    def get_cached_lora(self, layer_name: str) -> LoraPair | None:
        return self._cache.get(layer_name)

    # ------------------------------------------------------------------ regularizer helpers
    def generate_lora(self, conds: torch.Tensor) -> dict[str, LoraPair]:
        """Heads on a batch of conditionings with the CURRENT parameters (regularizer targets)."""
        cond = conds.to(next(self.parameters()).dtype)
        return {name: self.heads[_key(name)](cond)[1:] for name in self.layer_names}

    def lora_from_params(self, conds: torch.Tensor,
                         params: dict[str, torch.Tensor]) -> dict[str, LoraPair]:
        """Heads at overridden parameters (the lookahead phi + delta phi), functionally."""
        from torch.func import functional_call
        cond = conds.to(next(self.parameters()).dtype)
        out: dict[str, LoraPair] = {}
        for name in self.layer_names:
            k = _key(name)
            sub = {pn[len(k) + 1:]: pv for pn, pv in params.items() if pn.startswith(k + ".")}
            _, x_L, x_R = functional_call(self.heads[k], sub, (cond,))
            out[name] = (x_L, x_R)
        return out

    def hyper_parameters(self) -> list[nn.Parameter]:
        params = list(self.heads.parameters())
        params = params + (list(self.ground_head.parameters()) if self.ground_head is not None else [])
        if self.ground_gsa_mods is not None:
            params = params + list(self.ground_gsa_mods.parameters()) + list(self.ground_film.parameters())
        return params + (list(self.ground_gates.parameters()) if self.ground_gates is not None else [])

    def localization_parameters(self) -> list[nn.Parameter]:
        """Token head, gates, read projections, FiLM head: trained by the fit term only."""
        if self.ground_head is None:
            return []
        return (list(self.ground_head.parameters()) + list(self.ground_gates.parameters())
                + (list(self.ground_gsa_mods.parameters()) + list(self.ground_film.parameters())
                   if self.ground_gsa_mods is not None else []))


def build_hyper(bundle, rank: int = 4, head_hidden: int = 50, alpha_init: float = 1.0,
                learn_alpha: bool = True, target_modules: tuple[str, ...] = DEFAULT_TARGETS,
                n_tasks: int = 0, task_cond: dict | None = None) -> Hypernetwork:
    """Build the heads, inject the LoRA wrappers and attach the hypernetwork to the UNet."""
    cond_dim = int((task_cond or {}).get("key_dim") or bundle.clip_hidden_size)
    manager = Hypernetwork(n_tasks=n_tasks, cond_dim=cond_dim, task_cond=task_cond,
                                    clip_dim=int(bundle.clip_hidden_size))
    wrappers = inject_lora(bundle.unet, target_modules)
    if not wrappers:
        raise RuntimeError(f"inject_lora matched 0 modules for targets {target_modules}")
    print(f"[hyper] {len(wrappers)} LoRA layers for targets {list(target_modules)}", flush=True)
    for name, wrapper in wrappers:
        head = HyperHead(in_dim=wrapper.in_features, out_dim=wrapper.out_features,
                         cond_dim=cond_dim, rank=rank, hidden=head_hidden,
                         alpha_init=alpha_init, learn_alpha=learn_alpha)
        manager.add_head(name, head)
        wrapper.set_parent(bundle.unet)
    if manager.ground_cond:
        manager.init_ground_gates(manager.layer_names)
        if manager.ground_gsa:
            dims = {}
            for n, mod in bundle.unet.named_modules():
                if n.endswith("attn2.to_q") and hasattr(mod, "in_features"):
                    dims[n] = int(mod.in_features)
            manager.init_ground_gsa(dims)
    bundle.unet.hyper = manager
    manager.to(bundle.device)
    manager.float()   # the hypernetwork stays fp32 above a lower-precision backbone
    return manager
