"""Placement test of Table (grounding): one concept per generation, asked for each quadrant.

For every object concept (styles are skipped), every quadrant box and `--n` seeds, the prompt
"a photo of <class> <scene>" is generated and scored:
  quadrant  -- the quadrant crop most similar to the references (DINO) is the requested one
  IoU       -- Mask R-CNN detection vs the requested box; also the share with IoU > 0.5
  contain.  -- fraction of the detected mask inside the box
  fill      -- detected box area / requested box area
  TA        -- 2.5 * max(0, cos(image, prompt)) with CLIP ViT-B/32
  DINO_mask -- identity on the detected subject's pixels (background grayed out)
  color     -- distance of the subject's mean color to the reference cutouts' mean color
The detection picks the hinted COCO class when present, otherwise the candidate most similar to
the references under DINO; below `--dino_floor` the generation counts as undetected.

Rows of the table (all at s_lora = 0.7, 50 steps, "on a beach", 3 seeds x 4 boxes x 7 objects):
  --grid 0:0                                    no box (branch silent)
  --grid 2:0.3 --layout prompt                  position named in the prompt
  --grid 2:0.3 --layout regional                training-free layout, same adapters
  --grid 1:1                                    localization module, kappa = 1
  --grid 2:0.3                                  localization module, kappa = 2
  --grid 2:0.3 --bootstrap 15 --scaffold_steps 10    + bootstrapping (default setting)
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageDraw

from .backbone import load_sd, load_sdxl
from .common import load_config, load_hyper
from .data import token_span_mask
from .hypernet import build_hyper
from .injection import DEFAULT_TARGETS
from .localization import set_grounded
from .metrics import _Clip, _Dino
from .sampling import ddim_sample

BOXES = {"TL": (0.25, 0.25, 0.5, 0.5), "TR": (0.75, 0.25, 0.5, 0.5),
         "BL": (0.25, 0.75, 0.5, 0.5), "BR": (0.75, 0.75, 0.5, 0.5)}
POS_PHRASE = {"TL": "in the top left", "TR": "in the top right",
              "BL": "in the bottom left", "BR": "in the bottom right"}
# COCO classes hinted per concept (the rubber duck has no class of its own)
HINT = {"cifc_dog": ("dog",), "cifc_dog2": ("dog",), "cifc_cat": ("cat",),
        "cifc_cat2": ("cat",), "cifc_backpack": ("backpack", "handbag"),
        "cifc_teddybear": ("teddy bear",), "cifc_duck_toy": ("bird", "teddy bear")}
KEYS = ("iou", "hit", "con", "fill", "q", "n", "ndet", "dino", "dcol", "ncol",
        "dcrop", "dmaskd", "ncrop", "ta")


class ConfineAttnProcessor:
    """Training-free layout baseline: positions outside the box cannot attend to the concept's
    tokens (logit penalty 1e4), in the conditional pass only. Replaces the grounded processor."""

    def __init__(self, box_xyxy, token_mask, manager):
        self.box = box_xyxy
        self.tm = token_mask
        self.manager = manager
        self._cache = {}

    def _bias(self, n_img, n_tok, device, dtype):
        key = (n_img, n_tok, str(device), str(dtype))
        if key in self._cache:
            return self._cache[key]
        side = int(round(n_img ** 0.5))
        bias = torch.zeros(n_img, n_tok, device=device, dtype=torch.float32)
        if side * side == n_img:
            x0, y0, x1, y1 = self.box
            m = torch.zeros(side, side, device=device)
            c0, c1 = int(x0 * side), max(int(x0 * side) + 1, int(round(x1 * side)))
            r0, r1 = int(y0 * side), max(int(y0 * side) + 1, int(round(y1 * side)))
            m[r0:r1, c0:c1] = 1.0
            inside = m.reshape(-1)
            mine = self.tm.reshape(-1)[:n_tok].to(device).float()
            bias = bias - torch.outer(1.0 - inside, mine) * 1e4
        bias = bias.to(dtype)
        self._cache[key] = bias
        return bias

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
        q, k, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        n_img = q.shape[1]
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)
        cond_pass = bool(getattr(self.manager, "lora_enabled", True))
        bias = None
        if encoder_hidden_states is not None and cond_pass:
            bias = self._bias(n_img, k.shape[1], q.device, q.dtype)
        scores = torch.baddbmm(
            torch.zeros(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=attn.scale)
        if bias is not None:
            scores = scores + bias
        hidden_states = torch.bmm(scores.softmax(dim=-1).to(v.dtype), v)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def set_confine(unet, box_xyxy, token_mask, manager) -> None:
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and hasattr(mod, "set_processor"):
            mod.set_processor(ConfineAttnProcessor(box_xyxy, token_mask, manager))


def to_xyxy(box, W, H):
    cx, cy, bw, bh = box
    return ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)


def iou_parts(p, q):
    ix0, iy0 = max(p[0], q[0]), max(p[1], q[1])
    ix1, iy1 = min(p[2], q[2]), min(p[3], q[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ap_ = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
    aq = max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1])
    union = ap_ + aq - inter
    return (inter / union if union > 0 else 0.0), inter, ap_


def to_pil(t):
    return Image.fromarray((t.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())


def crops(img):
    H, W = img.shape[-2:]
    return {"TL": img[:, :H // 2, :W // 2], "TR": img[:, :H // 2, W // 2:],
            "BL": img[:, H // 2:, :W // 2], "BR": img[:, H // 2:, W // 2:]}


def main():
    ap = argparse.ArgumentParser(description="detector-based placement test")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--grid", default="2:0.3", help="kappa:sched, comma-separated")
    ap.add_argument("--layout", default="branch", choices=["branch", "prompt", "regional"],
                    help="branch = localization module; prompt = position named in the prompt, "
                         "branch silent; regional = training-free confinement, branch silent")
    ap.add_argument("--n", type=int, default=3, help="seeds per concept and box")
    ap.add_argument("--scale", type=float, default=0.7, help="s_lora")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--det_thr", type=float, default=0.3)
    ap.add_argument("--dino_floor", type=float, default=0.35)
    ap.add_argument("--seg_dir", default="data/seg", help="reference cutouts, for the color")
    ap.add_argument("--scene", default="on a beach")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrapping steps K")
    ap.add_argument("--scaffold_steps", type=int, default=0,
                    help="steps of the background latent generated in the same call, from the "
                         "prompt with the concept removed and both modules off")
    ap.add_argument("--seed0", type=int, default=31337)
    ap.add_argument("--only_concepts", default="",
                    help="comma-separated concept_ids; empty = all object concepts")
    ap.add_argument("--out", default="", help="directory for previews with boxes (empty = none)")
    a = ap.parse_args()
    if a.bootstrap > 0 and a.scaffold_steps <= 0:
        raise SystemExit("--bootstrap needs --scaffold_steps (the background latent)")

    grid = [tuple(float(x) for x in p.split(":"))[:2] for p in a.grid.split(",")]
    cfg = load_config(a.config)
    mid = cfg.get("sd_model_id", "")
    if "xl" in str(mid).lower():
        bundle = load_sdxl(model_id=mid, device="cuda", dtype=torch.float16)
    else:
        bundle = load_sd(device="cuda", dtype=torch.float16)
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    load_hyper(manager, a.ckpt, map_location="cuda")
    manager.eval()
    manager.lora_scale = a.scale
    set_grounded(bundle.unet, manager)
    dino = _Dino("cuda")
    clip = _Clip("cuda")

    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    _w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    det = maskrcnn_resnet50_fpn_v2(weights=_w).eval().to("cuda")
    cats = _w.meta["categories"]
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    def ref_color(concept_id):
        tot, wsum = np.zeros(3), 0.0
        for p in sorted(glob.glob(os.path.join(a.seg_dir, concept_id, "*.png"))):
            arr = np.asarray(Image.open(p).convert("RGBA"), dtype=np.float32) / 255.0
            m = arr[..., 3] > 0.5
            if not m.any():
                continue
            tot += arr[..., :3][m].sum(0)
            wsum += float(m.sum())
        return (tot / wsum) if wsum > 0 else None

    @torch.no_grad()
    def dino_feat(pil):
        return Fn.normalize(dino.m(dino.tf(pil).unsqueeze(0).to("cuda")), dim=-1)

    @torch.no_grad()
    def detect(img, pil, hints, ref):
        out = det([img.float()])[0]
        keep = out["scores"] >= a.det_thr
        boxes, labels, masks = out["boxes"][keep], out["labels"][keep], out["masks"][keep]
        scores = out["scores"][keep]
        if len(boxes) == 0:
            return None, None
        hid = [cats.index(x) for x in (hints or ()) if x in cats]
        if hid:
            sel = torch.isin(labels, torch.tensor(hid, device=labels.device)).nonzero().flatten()
            if len(sel) > 0:
                i = int(sel[int(scores[sel].argmax())])
                return tuple(float(v) for v in boxes[i]), masks[i, 0] > 0.5
        best, bsim = None, -1.0
        for i in range(len(boxes)):
            x0, y0, x1, y1 = [int(v) for v in boxes[i]]
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            sim = float((dino_feat(pil.crop((x0, y0, x1, y1))) @ ref.t()).item())
            if sim > bsim:
                best, bsim = i, sim
        if best is None or bsim < a.dino_floor:
            return None, None
        return tuple(float(v) for v in boxes[best]), masks[best, 0] > 0.5

    for kap, sched in grid:
        kap_eff = kap if a.layout == "branch" else 0.0
        manager.ground_gain_base = kap_eff
        manager.ground_sched_frac = sched
        print(f"=== kappa={kap_eff} sched={sched} layout={a.layout}", flush=True)
        agg = dict.fromkeys(KEYS, 0.0)
        for j, c in enumerate(cfg["concepts"]):
            if c.get("category") == "style":
                continue
            if a.only_concepts and c["concept_id"] not in a.only_concepts.split(","):
                continue
            ref = dino.img_feats(sorted(glob.glob(os.path.join(c["images_dir"], "*")))).mean(0, keepdim=True)
            rcol = ref_color(c["concept_id"])
            cls = c["class_word"]
            prompt = "a photo of " + cls + ((" " + a.scene.strip()) if a.scene else "")
            txt = clip.txt_feats([prompt])
            ch, pooled, _ = bundle.encode_text([prompt])
            uh, up, _ = bundle.encode_text([""])
            tspan = token_span_mask(bundle.tokenizer, [prompt], cls).cuda()
            tm = tspan if cfg.get("token_mask_lora") else None
            st = dict.fromkeys(KEYS, 0.0)
            scaffold = {}

            @torch.no_grad()
            def scaffold_latent(i_seed):
                """Background from the prompt with the concept removed, both modules off."""
                if i_seed in scaffold:
                    return scaffold[i_seed]
                bgp = " ".join((prompt.replace(cls, " ")).split())
                chb, pooledb, _ = bundle.encode_text([bgp])
                keep = (manager.lora_scale, manager.cond_box, manager.ground_gain_base)
                manager.lora_scale, manager.cond_box, manager.ground_gain_base = 0.0, None, 0.0
                gb = torch.Generator(device="cuda").manual_seed(a.seed0 + i_seed)
                scaf = ddim_sample(bundle, manager, chb, uh, pooledb,
                                   num_inference_steps=a.scaffold_steps, guidance_scale=7.5,
                                   generator=gb, task_idx=j, token_mask=None, uncond_pooled=up)[0]
                manager.lora_scale, manager.cond_box, manager.ground_gain_base = keep
                scaffold[i_seed] = bundle.encode_images(scaf.unsqueeze(0) * 2 - 1)
                return scaffold[i_seed]

            for bname, box in BOXES.items():
                manager.cond_box = box
                bch, bpooled, btm = ch, pooled, tm
                if a.layout == "prompt":
                    bprompt = prompt + " " + POS_PHRASE[bname]
                    bch, bpooled, _ = bundle.encode_text([bprompt])
                    bspan = token_span_mask(bundle.tokenizer, [bprompt], cls).cuda()
                    btm = bspan if cfg.get("token_mask_lora") else None
                if a.layout == "regional":
                    cx, cy, bw, bh = box
                    set_confine(bundle.unet, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                                tspan, manager)
                for i in range(a.n):
                    g = torch.Generator(device="cuda").manual_seed(a.seed0 + i)
                    img = ddim_sample(bundle, manager, bch, uh, bpooled, num_inference_steps=a.steps,
                                      guidance_scale=7.5, generator=g, task_idx=j, token_mask=btm,
                                      uncond_pooled=up, bootstrap_steps=a.bootstrap,
                                      bootstrap_bg=(scaffold_latent(i) if a.scaffold_steps else None))[0]
                    H, W = img.shape[-2:]
                    pil = to_pil(img)
                    sims = {kk: float((dino_feat(to_pil(v)) @ ref.t()).item())
                            for kk, v in crops(img).items()}
                    st["q"] += int(max(sims, key=sims.get) == bname)
                    st["dino"] += float((dino_feat(pil) @ ref.t()).item())
                    fi = clip.img_feats_from_tensor(img.unsqueeze(0).float())
                    st["ta"] += float((2.5 * (fi * txt).sum(-1).clamp(min=0)).mean().item())
                    st["n"] += 1
                    req = to_xyxy(box, W, H)
                    dbox, dmask = detect(img, pil, HINT.get(c["concept_id"]), ref)
                    if dbox is not None:
                        iou, inter, adet = iou_parts(dbox, req)
                        abox = (req[2] - req[0]) * (req[3] - req[1])
                        st["ndet"] += 1
                        st["iou"] += iou
                        st["hit"] += int(iou > 0.5)
                        st["fill"] += adet / abox if abox > 0 else 0.0
                        inb = torch.zeros_like(dmask)
                        inb[int(req[1]):int(req[3]), int(req[0]):int(req[2])] = True
                        tot = float(dmask.sum())
                        st["con"] += float((dmask & inb).sum()) / tot if tot > 0 else 0.0
                        if rcol is not None and tot > 0:
                            gc = img.permute(1, 2, 0)[dmask].clamp(0, 1).float().cpu().numpy().mean(0)
                            st["dcol"] += float(np.linalg.norm(gc - rcol))
                            st["ncol"] += 1
                        x0, y0, x1, y1 = [int(v) for v in dbox]
                        if x1 - x0 > 8 and y1 - y0 > 8:
                            st["dcrop"] += float((dino_feat(pil.crop((x0, y0, x1, y1))) @ ref.t()).item())
                            obj = img.clone()
                            obj[:, ~dmask] = 0.5
                            st["dmaskd"] += float((dino_feat(to_pil(obj).crop((x0, y0, x1, y1)))
                                                   @ ref.t()).item())
                            st["ncrop"] += 1
                    if a.out and i == 0:
                        dr = ImageDraw.Draw(pil)
                        dr.rectangle(req, outline=(255, 0, 0), width=3)
                        if dbox is not None:
                            dr.rectangle(dbox, outline=(0, 255, 0), width=3)
                        stem = f"{c['concept_id']}_k{kap_eff}_s{sched}_{bname}_{a.layout}"
                        pil.save(os.path.join(a.out, stem + ".png"))
            if a.layout == "regional":
                set_grounded(bundle.unet, manager)
            n, nd = max(1.0, st["n"]), max(1.0, st["ndet"])
            print(f"  {c['concept_id']:<16} quadrant {st['q']/n:.0%} | IoU {st['iou']/nd:.3f} | "
                  f"IoU>0.5 {st['hit']/nd:.0%} | contain {st['con']/nd:.2f} | fill {st['fill']/nd:.2f} | "
                  f"TA {st['ta']/n:.4f} | DINO_mask {st['dmaskd']/max(1.0, st['ncrop']):.4f} | "
                  f"det {int(st['ndet'])}/{int(st['n'])}", flush=True)
            for kk in KEYS:
                agg[kk] += st[kk]
        n, nd = max(1.0, agg["n"]), max(1.0, agg["ndet"])
        print(f"  TOTAL            quadrant {agg['q']/n:.0%} | IoU {agg['iou']/nd:.3f} | "
              f"IoU>0.5 {agg['hit']/nd:.0%} | contain {agg['con']/nd:.2f} | fill {agg['fill']/nd:.2f} | "
              f"TA {agg['ta']/n:.4f} | DINO {agg['dino']/n:.4f} | "
              f"DINO_crop {agg['dcrop']/max(1.0, agg['ncrop']):.4f} | "
              f"DINO_mask {agg['dmaskd']/max(1.0, agg['ncrop']):.4f} | "
              f"color {agg['dcol']/max(1.0, agg['ncol']):.3f} | "
              f"det {int(agg['ndet'])}/{int(agg['n'])} | layout {a.layout}", flush=True)
    print("PLACEMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
