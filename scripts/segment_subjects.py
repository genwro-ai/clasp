"""Subject cutouts for the paste composites: ISNet (via rembg) on every training photo of every
object concept, tight RGBA crop with a hardened, lightly feathered alpha -> data/seg/<concept_id>/.
Style concepts are skipped.

  python -m scripts.segment_subjects --config configs/sd15_seq50.yaml --out data/seg
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove

from clasp.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/seg")
    ap.add_argument("--min_alpha_frac", type=float, default=0.05,
                    help="reject a segmentation covering less than this fraction of the image")
    a = ap.parse_args()
    cfg = load_config(a.config)
    sess = new_session("isnet-general-use")
    for c in cfg["concepts"]:
        if c.get("category") == "style":
            continue
        d = os.path.join(a.out, c["concept_id"])
        os.makedirs(d, exist_ok=True)
        paths = sorted(sum((glob.glob(os.path.join(c["images_dir"], e))
                            for e in ("*.jpg", "*.jpeg", "*.png")), []))
        kept = 0
        for p in paths:
            img = Image.open(p).convert("RGB")
            cut = remove(img, session=sess)
            arr = np.asarray(cut).copy()
            hard = (arr[:, :, 3] > 38).astype(np.uint8) * 255
            soft = np.asarray(Image.fromarray(hard).filter(ImageFilter.GaussianBlur(2)))
            arr[:, :, 3] = np.where(hard > 0, np.maximum(soft, 128), soft)
            cut = Image.fromarray(arr)
            al = np.asarray(cut)[:, :, 3]
            ys, xs = np.where(al > 16)
            if len(xs) == 0 or len(xs) < a.min_alpha_frac * al.size:
                print(f"[seg] rejected (empty mask): {p}", flush=True)
                continue
            cut = cut.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            cut.save(os.path.join(d, os.path.splitext(os.path.basename(p))[0] + ".png"))
            kept += 1
        print(f"[seg] {c['concept_id']}: {kept}/{len(paths)} cutouts -> {d}", flush=True)


if __name__ == "__main__":
    main()
