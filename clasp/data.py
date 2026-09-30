"""One concept's photographs with their per-image captions, and the token mask of the class word.

Captions are the benchmark's per-image captions (`caption/<concept>/<stem>.txt`). Phrases listed
under `attr_strip` (e.g. "red backpack") are replaced by the bare class word first, so no caption
names an attribute of the concept itself. If the config gives an identifier, the class word is
replaced by "<identifier> <class>"; the configs here use an empty identifier.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ConceptSpec:
    concept_id: str
    images_dir: str
    class_word: str
    identifier: str
    caption_dir: str | None = None
    attr_strip: list | None = None
    prompt: str | None = None
    category: str | None = None

    @property
    def replacement(self) -> str:
        return f"{self.identifier} {self.class_word}"

    @property
    def diag_prompt(self) -> str:
        return self.prompt or f"a photo of {self.replacement}"


def _load_image(path: str, resolution: int, augment: bool = False):
    """-> (tensor in [-1, 1] [3, H, W], (orig_h, orig_w), (crop_top, crop_left)).

    The source size and crop offset are SDXL's micro-conditioning; SD-1.5 ignores them.
    With `augment`: random square crop of 80-100% of the short side and a random flip."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if augment:
        s = int(min(w, h) * random.uniform(0.8, 1.0))
        x0 = random.randint(0, w - s)
        y0 = random.randint(0, h - s)
        img = img.crop((x0, y0, x0 + s, y0 + s))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        s = min(w, h)
        x0, y0 = (w - s) // 2, (h - s) // 2
        img = img.crop((x0, y0, x0 + s, y0 + s))
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    crop = (int(round(y0 * resolution / s)), int(round(x0 * resolution / s)))
    return arr * 2.0 - 1.0, (h, w), crop


class ConceptDataset(Dataset):
    def __init__(self, spec: ConceptSpec, resolution: int = 512, augment: bool = False):
        self.spec = spec
        self.resolution = resolution
        self.augment = augment
        self.paths = sorted(
            p for p in glob.glob(os.path.join(spec.images_dir, "*")) if p.lower().endswith(_IMG_EXTS))
        if not self.paths:
            raise FileNotFoundError(f"no images for concept '{spec.concept_id}' in {spec.images_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def _caption(self, stem: str) -> str:
        if self.spec.caption_dir:
            cap_path = os.path.join(self.spec.caption_dir, stem + ".txt")
            if os.path.exists(cap_path):
                with open(cap_path) as f:
                    cap = f.read().strip()
                repl, cls = self.spec.replacement, self.spec.class_word
                for ph in (self.spec.attr_strip or []):
                    cap = cap.replace(ph, cls)
                return cap.replace(cls, repl) if cls in cap else f"{repl}, {cap}"
        return self.spec.diag_prompt

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx % len(self.paths)]
        stem = os.path.splitext(os.path.basename(path))[0]
        px, orig, crop = _load_image(path, self.resolution, self.augment)
        return {"pixel_values": px, "caption": self._caption(stem),
                "orig_size": torch.tensor(orig), "crop": torch.tensor(crop)}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
        "captions": [b["caption"] for b in batch],
        "orig_size": torch.stack([b["orig_size"] for b in batch], 0),   # [B, 2] (h, w) of source
        "crop": torch.stack([b["crop"] for b in batch], 0),             # [B, 2] (top, left)
    }


def specs_from_config(concept_cfgs: list[dict]) -> list[ConceptSpec]:
    specs = []
    for i, c in enumerate(concept_cfgs):
        specs.append(ConceptSpec(
            concept_id=c["concept_id"],
            images_dir=c["images_dir"],
            class_word=c.get("class_word", c["concept_id"]),
            identifier=c.get("identifier", f"V{i + 1}"),
            attr_strip=c.get("attr_strip"),
            caption_dir=c.get("caption_dir"),
            prompt=c.get("prompt"),
            category=c.get("category"),
        ))
    return specs


def token_span_mask(tokenizer, prompts, phrase, max_length=None):
    """[B, L] mask: 1.0 at the sub-token positions of `phrase` in each prompt (all occurrences).
    A prompt that does not contain the phrase gets an all-ones row."""
    max_length = max_length or tokenizer.model_max_length
    enc = tokenizer(prompts, padding="max_length", max_length=max_length,
                    truncation=True)["input_ids"]
    pat = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    mask = torch.zeros(len(prompts), max_length)
    if not pat:
        return mask + 1.0
    for b, ids in enumerate(enc):
        for i in range(max_length - len(pat) + 1):
            if ids[i:i + len(pat)] == pat:
                mask[b, i:i + len(pat)] = 1.0
        if mask[b].sum() == 0:
            mask[b] = 1.0
    return mask
