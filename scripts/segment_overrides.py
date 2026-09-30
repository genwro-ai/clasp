"""Replace the automatic cutouts listed in data/mask_overrides.txt.

For nineteen photographs of the fifty-concept sequence and twenty-six of the concepts the
hundred-concept sequence adds, the automatic Grounded-SAM mask (scripts/segment_gsam.py) was wrong
or empty: a fragment of the background, the person wearing the object instead of the object, a
glass plate or a sofa full of holes, or one object of a concept made of several. For each of them
we generated candidates over several prompts, the best Grounding-DINO boxes and the three SAM
masks, inspected them, and recorded the chosen one. This script regenerates exactly that choice,
so no inspection is needed to reproduce the masks.

Line format (whitespace separated, '#' starts a comment):
    <concept_id> <image file> <prompt, dots allowed, no spaces: '_' for a space> <box rank> <sam mask>
    <variant>
box rank: 0, 1, ... (boxes by Grounding-DINO score), or U: the union of the chosen SAM mask over
the three best boxes, for concepts made of several objects (a pair of dice, a table with chairs).
variant: raw | fill (holes filled) | fill1024 (holes filled on a copy downscaled to a long side of
1024 px, the procedure used for the concepts added by the hundred-concept sequence) | hull (convex
hull, for plates whose rim is broken or cut by the frame edge).
Lines whose concept is not in --config are skipped, so one file serves both sequences.

  python -m scripts.segment_overrides --config configs/sd15_seq50.yaml --out data/seg
  python -m scripts.segment_overrides --config configs/sd15_seq100.yaml --out data/seg
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.segment_gsam import detect, load_models, sam_masks, save_cut


def fill_holes(m):
    """Fill holes: flood the background from the border of a 1-px padded copy; everything the
    flood does not reach belongs to the object."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m.astype(np.uint8) * 255
    im = Image.fromarray(pad).copy()
    ImageDraw.floodfill(im, (0, 0), 128)
    return np.asarray(im)[1:-1, 1:-1] != 128


def fill_holes_1024(m):
    """fill_holes computed on a nearest-neighbour copy with a long side of 1024 px and scaled back,
    OR-ed with the original mask; identical to fill_holes for images that are not larger."""
    h, w = m.shape
    s = max(h, w) / 1024
    if s <= 1:
        return fill_holes(m)
    small = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (round(w / s), round(h / s)), Image.NEAREST)) > 0
    up = np.asarray(Image.fromarray(fill_holes_1024(small).astype(np.uint8) * 255).resize(
        (w, h), Image.NEAREST)) > 0
    return m | up


def convex_hull(m):
    """Convex hull of the mask (monotone chain over the row extremes), drawn as a filled polygon."""
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return m
    pts = []
    for y in np.unique(ys):
        row = xs[ys == y]
        pts += [(int(row.min()), int(y)), (int(row.max()), int(y))]
    pts = sorted(set(pts))

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    im = Image.new("L", (m.shape[1], m.shape[0]), 0)
    ImageDraw.Draw(im).polygon(lower[:-1] + upper[:-1], fill=255)
    return np.asarray(im) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", default="data/mask_overrides.txt")
    ap.add_argument("--config", default="configs/sd15_seq50.yaml")
    ap.add_argument("--out", default="data/seg")
    ap.add_argument("--box_thr", type=float, default=0.15)
    ap.add_argument("--text_thr", type=float, default=0.15)
    a = ap.parse_args()
    from clasp.common import load_config
    dirs = {c["concept_id"]: c["images_dir"] for c in load_config(a.config)["concepts"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_models(device)
    n = 0
    for line in open(a.overrides, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cid, fname, prompt, box_rank, mask_idx, variant = line.split()
        if cid not in dirs:
            continue
        prompt = prompt.replace("_", " ")
        img = Image.open(os.path.join(dirs[cid], fname)).convert("RGB")
        boxes, _ = detect(img, prompt, models, device, a.box_thr, a.text_thr)
        if box_rank == "U":
            assert len(boxes) >= 2, (cid, fname, prompt, len(boxes))
            m = np.logical_or.reduce([sam_masks(img, b, models, device)[0][int(mask_idx)] for b in boxes[:3]])
        else:
            assert len(boxes) > int(box_rank), (cid, fname, prompt, len(boxes))
            m = sam_masks(img, boxes[int(box_rank)], models, device)[0][int(mask_idx)]
        post = {"raw": lambda x: x, "fill": fill_holes, "fill1024": fill_holes_1024, "hull": convex_hull}
        m = post[variant](m)
        d = os.path.join(a.out, cid)
        os.makedirs(d, exist_ok=True)
        ok = save_cut(img, m, os.path.join(d, os.path.splitext(fname)[0] + ".png"), 0.01)
        assert ok, (cid, fname)
        n += 1
        print(f"[override] {cid}/{fname}: '{prompt}' box {box_rank} mask {mask_idx} {variant}", flush=True)
    print(f"[override] {n} cutouts replaced", flush=True)


if __name__ == "__main__":
    main()
