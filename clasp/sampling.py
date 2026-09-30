"""Sampling with classifier-free guidance and the generated adapter.

The update is computed once per generation (it does not depend on the timestep). The
conditional pass runs with the adapter and the localization branch on, the unconditional pass
with both off. `ground_gain_base` (kappa) and `ground_sched_frac` on the manager set the gain of
the branch and the fraction of steps it is active for (defaults 1.0 and 1.0).

Bootstrapping (optional): for the first K steps everything outside the dilated box is replaced by
a background latent noised to the current level, so the subject can only form inside the box.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def ddim_sample(bundle, manager, cond_hidden: torch.Tensor, uncond_hidden: torch.Tensor,
                clip_pooled: torch.Tensor, num_inference_steps: int = 50,
                guidance_scale: float = 7.5, height: int = 512, width: int = 512,
                batch_size: int = 1, generator: torch.Generator | None = None,
                scheduler=None, task_idx: int | None = None,
                token_mask: torch.Tensor | None = None,
                latents: torch.Tensor | None = None,
                bootstrap_steps: int = 0,
                bootstrap_bg: torch.Tensor | None = None,
                uncond_pooled: torch.Tensor | None = None) -> torch.Tensor:
    """Returns images in [0, 1], [batch_size, 3, H, W]."""
    device, dtype = bundle.device, bundle.dtype
    unet = bundle.unet
    scheduler = scheduler if scheduler is not None else bundle.ddim_scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    lh, lw = height // 8, width // 8
    if latents is None:
        latents = torch.randn(batch_size, bundle.latent_channels, lh, lw,
                              generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    cond_seq = cond_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
    ac_c = bundle.added_cond(batch_size, height, width, pooled=clip_pooled)
    ac_u = dict(ac_c)
    if ac_c:
        # SDXL: the unconditional branch gets the negative prompt's own pooled embedding
        if uncond_pooled is not None:
            ac_u = bundle.added_cond(batch_size, height, width, pooled=uncond_pooled)
        else:
            ac_u = {**ac_c, "text_embeds": torch.zeros_like(ac_c["text_embeds"])}
    uncond_seq = uncond_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    if getattr(manager, "ground_cond", False):
        manager.set_ground(task_idx, getattr(manager, "cond_box", None))
    manager.set_context(clip_pooled.to(device), task_idx=task_idx,
                        token_mask=token_mask.to(device) if token_mask is not None else None)
    manager.compute_and_cache_loras()

    steps_list = list(scheduler.timesteps)
    bs_mask = None
    if bootstrap_steps > 0:
        if bootstrap_bg is None or getattr(manager, "cond_box", None) is None:
            raise ValueError("bootstrapping needs a box and a background latent")
        cx, cy, bw, bh = manager.cond_box
        # hard mask on the box dilated by `bs_dilate` latent cells (a soft blend of two noise
        # draws has too little variance for its timestep and desaturates the output)
        d = int(getattr(manager, "bs_dilate", 3))
        y0 = max(0, int((cy - bh / 2) * lh) - d)
        y1 = min(lh, max(y0 + 1, int(round((cy + bh / 2) * lh)) + d))
        x0 = max(0, int((cx - bw / 2) * lw) - d)
        x1 = min(lw, max(x0 + 1, int(round((cx + bw / 2) * lw)) + d))
        bs_mask = torch.zeros(1, 1, lh, lw, device=device, dtype=dtype)
        bs_mask[:, :, y0:y1, x0:x1] = 1.0
        bootstrap_bg = bootstrap_bg.to(device=device, dtype=dtype)

    _MISSING = object()
    _prev_gain = getattr(manager, "ground_gain", _MISSING)
    for i, t in enumerate(steps_list):
        if getattr(manager, "ground_cond", False):
            frac = i / max(1, len(steps_list))
            base = float(getattr(manager, "ground_gain_base", 1.0))
            sched = float(getattr(manager, "ground_sched_frac", 1.0))
            manager.ground_gain = base if frac < sched else 0.0
        manager.enable_lora()
        if bs_mask is not None and i < bootstrap_steps:
            noise_bg = torch.randn(bootstrap_bg.shape, generator=generator, device=device,
                                   dtype=dtype)
            bg_t = scheduler.add_noise(bootstrap_bg, noise_bg, t.reshape(1))
            latents = latents * bs_mask + bg_t.expand_as(latents) * (1 - bs_mask)
        model_input = scheduler.scale_model_input(latents, t)
        noise_cond = unet(model_input, t, encoder_hidden_states=cond_seq,
                          added_cond_kwargs=ac_c or None).sample
        with manager.no_lora():
            noise_uncond = unet(model_input, t, encoder_hidden_states=uncond_seq,
                                added_cond_kwargs=ac_u or None).sample
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if _prev_gain is _MISSING:
        if hasattr(manager, "ground_gain"):
            del manager.ground_gain
    else:
        manager.ground_gain = _prev_gain
    return bundle.decode_latents(latents)
