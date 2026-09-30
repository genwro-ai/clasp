"""TA, IA and DINO over the evaluation matrix, and forgetting.

The formulas and models follow the benchmark's evaluation script:
  * TA   : 2.5 * mean(max(0, cos(image, prompt text)))           [CLIP ViT-B/32]
  * IA   : mean over all generated x reference pairs of cos       [CLIP ViT-B/32]
  * DINO : mean over all generated x reference pairs of cos       [DINO ViT-S/16]
Forgetting per concept j is max_{j <= k < final} score(k, j) - score(final, j), averaged over j.
TA and IA are usually quoted as these values times 100.

Run: python -m clasp.metrics --config configs/sd15_bench10.yaml --eval_root outputs/.../s045
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import load_config


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# same extensions and case-insensitivity as data.ConceptDataset
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _imgs(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*"))
                  if p.lower().endswith(_IMG_EXTS))


class _Clip:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    @torch.no_grad()
    def img_feats(self, paths):
        out = []
        for b in _chunks(paths, 64):
            ims = [Image.open(p).convert("RGB") for p in b]
            inp = self.p(images=ims, return_tensors="pt").to(self.device)
            out.append(F.normalize(self.m.get_image_features(**inp), dim=-1))
        return torch.cat(out) if out else torch.empty(0)

    @torch.no_grad()
    def img_feats_from_tensor(self, imgs):
        """imgs: [N,3,H,W] in [0,1] -> L2-normalised CLIP image features (for memory banks)."""
        out = []
        for b in _chunks(list(range(imgs.shape[0])), 64):
            batch = imgs[b[0]:b[-1] + 1]
            pil = [Image.fromarray((x.permute(1, 2, 0).clamp(0, 1) * 255)
                                   .byte().cpu().numpy()) for x in batch]
            inp = self.p(images=pil, return_tensors="pt").to(self.device)
            out.append(F.normalize(self.m.get_image_features(**inp), dim=-1))
        return torch.cat(out) if out else torch.empty(0)

    @torch.no_grad()
    def txt_feats(self, texts):
        out = []
        for b in _chunks(texts, 64):
            inp = self.p(text=b, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out.append(F.normalize(self.m.get_text_features(**inp), dim=-1))
        return torch.cat(out) if out else torch.empty(0)


class _Dino:
    def __init__(self, device):
        import timm
        from torchvision import transforms
        self.m = timm.create_model("vit_small_patch16_224.dino", pretrained=True,
                                   num_classes=0).to(device).eval()
        # the benchmark evaluate.py preprocessing: Resize(256, bicubic) + CenterCrop(224) +
        # ImageNet norm (timm's auto config would use crop_pct=0.9, i.e. resize 249 -> crop 224)
        self.tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.device = device

    @torch.no_grad()
    def img_feats(self, paths):
        out = []
        for b in _chunks(paths, 64):
            x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in b]).to(self.device)
            out.append(F.normalize(self.m(x), dim=-1))
        return torch.cat(out) if out else torch.empty(0)


def _cross_cos(a, b):
    return float((a @ b.T).mean().item())


def _clip_t(img_feats, txt_feats):
    return float((2.5 * (img_feats * txt_feats).sum(-1).clamp(min=0)).mean().item())


def main():
    ap = argparse.ArgumentParser(description="TA / IA / DINO and forgetting over the matrix")
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_root", required=True, help="clasp.generate output (after_task{k}/...)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    concepts = cfg["concepts"]
    n = len(concepts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, dino = _Clip(device), _Dino(device)

    # reference features per concept (cached once)
    ref_clip, ref_dino = {}, {}
    for j, c in enumerate(concepts):
        rp = _imgs(c["images_dir"])
        if not rp:
            raise FileNotFoundError(
                f"concept {c['concept_id']}: no reference images in {c['images_dir']}")
        ref_clip[j], ref_dino[j] = clip.img_feats(rp), dino.img_feats(rp)

    M = {}
    for k in range(n):
        for j in range(k + 1):
            d = os.path.join(args.eval_root, f"after_task{k:02d}", f"task{j:02d}_{concepts[j]['concept_id']}")
            sdir, pj = os.path.join(d, "samples"), os.path.join(d, "prompts.json")
            paths = _imgs(sdir)
            if not paths or not os.path.exists(pj):
                continue
            gi_clip, gi_dino = clip.img_feats(paths), dino.img_feats(paths)
            with open(pj) as f:
                info = json.load(f)
            id2t = {list(x.keys())[0]: list(x.values())[0] for x in info}
            texts = [id2t.get(os.path.splitext(os.path.basename(p))[0], "") for p in paths]
            tf = clip.txt_feats(texts)
            M[(k, j)] = {"clip_t": _clip_t(gi_clip, tf),
                         "clip_i": _cross_cos(gi_clip, ref_clip[j]),
                         "dino_i": _cross_cos(gi_dino, ref_dino[j])}
            print(f"[metrics] after_task{k:02d}/{concepts[j]['concept_id']}: "
                  f"CLIP-T {M[(k,j)]['clip_t']:.3f} CLIP-I {M[(k,j)]['clip_i']:.3f} "
                  f"DINO {M[(k,j)]['dino_i']:.3f}", flush=True)

    last = n - 1
    avg = {m: float(np.mean([M[(last, j)][m] for j in range(n) if (last, j) in M]))
           for m in ("clip_t", "clip_i", "dino_i")}
    forget = {}
    for m in ("clip_i", "dino_i"):
        drops = []
        for j in range(n):
            seq = [M[(k, j)][m] for k in range(j, n) if (k, j) in M]
            if len(seq) >= 2:
                drops.append(max(seq[:-1]) - seq[-1])
        forget[m] = float(np.mean(drops)) if drops else 0.0

    result = {
        "average_final": avg,                       # mean over concepts, final model
        "forgetting": forget,                        # mean (peak - final) per concept; >0 = degradation
        "matrix": {f"{k},{j}": v for (k, j), v in M.items()},
        "concepts": [c["concept_id"] for c in concepts],
    }
    print("\n=== CIDM/CIFC RESULTS ===")
    print(f"Average (final): CLIP-T {avg['clip_t']:.4f} | CLIP-I {avg['clip_i']:.4f} "
          f"| DINO {avg['dino_i']:.4f}")
    print(f"Forgetting: CLIP-I {forget['clip_i']:+.4f} | DINO {forget['dino_i']:+.4f}")
    out = args.out or os.path.join(args.eval_root, "cifc_metrics.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[metrics] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
