"""Subject cutouts for the forty concepts added from CustomConcept101: Grounded-SAM
(Grounding-DINO proposes a box for the concept's class word, SAM turns it into a mask), tight
RGBA crop with the same alpha treatment as scripts/segment_subjects.py -> data/seg/<concept_id>/.

The benchmark's own seven objects keep their ISNet cutouts (scripts/segment_subjects.py); this
script only touches the concepts it is given, so run it with --only_prefix cc101_.
After it, scripts/segment_overrides.py replaces the cutouts listed in data/mask_overrides.txt.

  python -m scripts.segment_gsam --config configs/sd15_seq50.yaml --out data/seg --only_prefix cc101_

Grounding-DINO and SAM on GPU are not guaranteed to be bitwise deterministic, so a regenerated
cutout can differ from ours in isolated edge pixels.
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoProcessor, GroundingDinoForObjectDetection, SamModel, SamProcessor

from clasp.common import load_config

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_models(device):
    gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd = gd.to(device).eval()
    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
    return gd_proc, gd, sam_proc, sam


@torch.no_grad()
def detect(img, text, models, device, box_thr, text_thr):
    """Boxes for `text`, sorted by score (highest first)."""
    gd_proc, gd, _, _ = models
    gi = gd_proc(images=img, text=text, return_tensors="pt").to(device)
    det = gd_proc.post_process_grounded_object_detection(
        gd(**gi), gi.input_ids, box_threshold=box_thr, text_threshold=text_thr,
        target_sizes=[img.size[::-1]])[0]
    order = det["scores"].argsort(descending=True).tolist()
    return [det["boxes"][i].tolist() for i in order], [float(det["scores"][i]) for i in order]


@torch.no_grad()
def sam_masks(img, box, models, device):
    """The three SAM masks for one box as bool arrays [3, H, W], and their predicted IoUs."""
    _, _, sam_proc, sam = models
    si = sam_proc(img, input_boxes=[[box]], return_tensors="pt").to(device)
    so = sam(**si)
    masks = sam_proc.image_processor.post_process_masks(
        so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu())[0][0]
    return masks.numpy().astype(bool), so.iou_scores[0, 0].cpu()


def save_cut(img, mask, path, min_frac):
    """Feather the binary mask by 2 px (minimum 128 inside), crop tight, save RGBA.
    Returns False (and saves nothing) when the mask covers less than `min_frac` of the image."""
    m = mask.astype(np.uint8) * 255
    soft = np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(2)))
    alpha = np.where(m > 0, np.maximum(soft, 128), soft).astype(np.uint8)
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0 or len(xs) < min_frac * alpha.size:
        return False
    arr = np.dstack([np.asarray(img), alpha])
    Image.fromarray(arr, "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).save(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/seg")
    ap.add_argument("--only_prefix", default="cc101_", help="segment only concept_ids with this prefix")
    ap.add_argument("--box_thr", type=float, default=0.3)
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--min_alpha_frac", type=float, default=0.05)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    for c in load_config(a.config)["concepts"]:
        if c.get("category") == "style" or not c["concept_id"].startswith(a.only_prefix):
            continue
        d = os.path.join(a.out, c["concept_id"])
        os.makedirs(d, exist_ok=True)
        text = c["class_word"].lower().strip().rstrip(".") + "."
        paths = sorted(sum((glob.glob(os.path.join(c["images_dir"], e)) for e in EXTS), []))
        kept = 0
        for p in paths:
            img = Image.open(p).convert("RGB")
            boxes, _ = detect(img, text, models, device, a.box_thr, a.text_thr)
            if not boxes:
                print(f"[gsam] no detection ({text}): {p} -> needs an override", flush=True)
                continue
            masks, iou = sam_masks(img, boxes[0], models, device)
            out = os.path.join(d, os.path.splitext(os.path.basename(p))[0] + ".png")
            if save_cut(img, masks[int(iou.argmax())], out, a.min_alpha_frac):
                kept += 1
            else:
                print(f"[gsam] empty mask: {p} -> needs an override", flush=True)
        print(f"[gsam] {c['concept_id']}: {kept}/{len(paths)} cutouts -> {d}", flush=True)


if __name__ == "__main__":
    main()
