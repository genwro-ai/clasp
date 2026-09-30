"""Backgrounds for the subject-paste composites: 20 empty scenes x 5 seeds from the frozen backbone,
no adapter. Backgrounds must come from the backbone and resolution used for training.

  python -m scripts.make_backgrounds --out data/backgrounds                          # SD-1.5, 512
  python -m scripts.make_backgrounds --out data/backgrounds_sdxl --sdxl --size 1024  # SDXL
"""
import argparse
import os
from contextlib import contextmanager

import torch
from torchvision.utils import save_image

from clasp.backbone import load_sd, load_sdxl
from clasp.sampling import ddim_sample

SCENES = [
    "a sandy beach with gentle waves, empty",
    "a green meadow with wildflowers, empty",
    "a forest clearing with soft sunlight",
    "a quiet city street with cobblestones, empty",
    "a cozy living room interior, empty floor",
    "a wooden kitchen table by a window",
    "a park lawn with trees in the background",
    "a snowy field under an overcast sky",
    "a desert landscape with distant dunes",
    "a stone patio in a garden, empty",
    "a lakeside shore at golden hour, empty",
    "a minimalist studio with plain backdrop",
    "a rustic barn interior with hay, empty",
    "a mountain trail with rocks and grass",
    "a library room with bookshelves, empty floor",
    "an autumn park with fallen leaves, empty",
    "a brick wall alley with soft light, empty",
    "a bathroom with tiled floor, empty",
    "a wooden pier over calm water, empty",
    "a grassy hill under a blue sky, empty",
]
PER_SCENE = 5
NEG = "person, animal, object in foreground, text, watermark"


class _NoLoRA:
    """Stand-in manager: plain backbone sampling."""
    lora_enabled = False
    lora_scale = 1.0
    def get_cached_lora(self, name): return None
    def get_token_mask(self): return None
    def enable_lora(self): pass
    @contextmanager
    def no_lora(self):
        yield
    def set_context(self, *a, **kw): pass
    def compute_and_cache_loras(self, *a, **kw): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/backgrounds")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--sdxl", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    b = load_sdxl(device=dev, dtype="fp16") if a.sdxl else load_sd(device=dev, dtype="fp16")
    side = a.size // 8
    mgr = _NoLoRA()
    for si, scene in enumerate(SCENES):
        ctx, pooled, _ = b.encode_text([scene])
        unc, _, _ = b.encode_text([NEG])
        lat = torch.stack([torch.randn((b.latent_channels, side, side),
                           generator=torch.Generator(device=dev).manual_seed(5000 + si * 100 + j),
                           device=dev, dtype=b.dtype) for j in range(PER_SCENE)])
        imgs = ddim_sample(b, mgr, ctx, unc, pooled, num_inference_steps=a.steps,
                           guidance_scale=7.5, batch_size=PER_SCENE, scheduler=b.dpm_scheduler,
                           height=a.size, width=a.size, latents=lat)
        for j in range(PER_SCENE):
            save_image(imgs[j], os.path.join(a.out, f"bg_{si:02d}_{j}.jpg"))
        print(f"[bg] {si + 1}/{len(SCENES)}: {scene}", flush=True)


if __name__ == "__main__":
    main()
