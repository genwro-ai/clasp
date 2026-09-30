"""Evaluation images for the forgetting matrix (box-free protocol).

For every checkpoint k (model after task k) and every concept j <= k, the benchmark's evaluation
prompts of the concept's category are generated with <TOK> replaced by the class word,
`--num_samples` images per prompt. Layout:
  <out_root>/after_task{k:02d}/task{j:02d}_{concept_id}/{samples/*.jpg, prompts.json}
which `clasp.metrics` scores.

Run:
  python -m clasp.generate --config configs/sd15_bench10.yaml \
      --ckpt_dir outputs/sd15_bench10/ckpts --out_root outputs/sd15_bench10/matrix/s045 \
      --num_samples 10 --lora_scale 0.45 --eval_dtype fp16
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torchvision.utils import save_image

from .backbone import load_backbone
from .common import load_config, load_hyper
from .data import token_span_mask
from .hypernet import build_hyper
from .injection import DEFAULT_TARGETS
from .localization import set_grounded
from .sampling import ddim_sample

CAT_FILE = {"pet": "test_pet.txt", "plushy": "test_plushy.txt", "style": "test_style.txt",
            "object": "test_object.txt"}
# the benchmark's prompts live in its repository; test_object.txt (for the CustomConcept101
# objects of the fifty-concept sequence) is in this repository, under data/prompts
PROMPT_DIRS = ("data/CIFC/datasets/evaluation_prompts", "data/prompts")
NEG = ("longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, "
       "worst quality, low quality")


def _read_prompts(category, prompt_dirs=PROMPT_DIRS):
    for d in prompt_dirs:
        p = os.path.join(d, CAT_FILE[category])
        if os.path.exists(p):
            with open(p) as f:
                return [ln.strip() for ln in f if ln.strip()]
    raise FileNotFoundError(f"no prompt file {CAT_FILE[category]} in {prompt_dirs}")


@torch.no_grad()
def gen_concept(bundle, manager, gen_repl, clipt_repl, category, out_dir, n, steps, gscale, seed,
                task_idx=None, mask_phrase=None, sample_batch=1):
    prompts = _read_prompts(category)
    sdir = os.path.join(out_dir, "samples")
    os.makedirs(sdir, exist_ok=True)
    uncond_hidden, uncond_pooled, _ = bundle.encode_text([NEG])
    res = int(bundle.default_resolution)
    lat_shape = (bundle.latent_channels, res // 8, res // 8)
    info, count = [], 0
    for p in prompts:
        gen_prompt = p.replace("<TOK>", gen_repl)
        clipt_text = p.replace("<TOK>", clipt_repl)
        cond_hidden, pooled, _ = bundle.encode_text([gen_prompt])
        token_mask = (token_span_mask(bundle.tokenizer, [gen_prompt], mask_phrase)
                      if mask_phrase else None)
        done = 0
        while done < n:
            bs = min(sample_batch, n - done)
            # one generator per image, so the images do not depend on sample_batch
            lat = torch.stack([
                torch.randn(lat_shape, generator=torch.Generator(device=bundle.device)
                            .manual_seed(seed + count + i), device=bundle.device,
                            dtype=bundle.dtype)
                for i in range(bs)])
            imgs = ddim_sample(bundle, manager, cond_hidden, uncond_hidden, pooled,
                               num_inference_steps=steps, guidance_scale=gscale, batch_size=bs,
                               height=res, width=res, scheduler=bundle.dpm_scheduler,
                               task_idx=task_idx, token_mask=token_mask, latents=lat,
                               uncond_pooled=uncond_pooled)
            for i in range(bs):
                save_image(imgs[i], os.path.join(sdir, f"{count}.jpg"))
                info.append({str(count): clipt_text})
                count += 1
            done += bs
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump(info, f)
    return count


def parse_args():
    p = argparse.ArgumentParser(description="evaluation images for the forgetting matrix")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_dir", required=True, help="directory with hyper_after_task{k:02d}.pt")
    p.add_argument("--out_root", required=True)
    p.add_argument("--num_samples", type=int, default=10, help="images per prompt")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--lora_scale", type=float, default=1.0, help="s_lora")
    p.add_argument("--lora_scale_map", default=None,
                   help='separate scales per projection group, e.g. '
                        '"attn2.to_out.0=0.5,attn2.to_q=0.2,attn2.to_k=0.2,attn2.to_v=0.2"; '
                        'first match wins, fallback --lora_scale (used for SDXL)')
    p.add_argument("--eval_dtype", default=None, choices=["fp32", "fp16", "bf16"],
                   help="backbone dtype for generation (the hypernetwork stays fp32)")
    p.add_argument("--sample_batch", type=int, default=10)
    p.add_argument("--final_only", action="store_true",
                   help="only the last checkpoint (one row of the matrix)")
    p.add_argument("--only_tasks", default=None,
                   help="comma-separated checkpoint indices k to generate")
    p.add_argument("--only_concepts", default=None,
                   help="comma-separated concept_ids to generate")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_backbone(cfg, device, dtype=args.eval_dtype)
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    if manager.ground_cond:
        set_grounded(bundle.unet, manager)
    manager.eval()
    manager.lora_scale = float(args.lora_scale)
    if args.lora_scale_map:
        manager.lora_scale_map = [(kv.split("=")[0], float(kv.split("=")[1]))
                                  for kv in args.lora_scale_map.split(",")]

    concepts = cfg["concepts"]
    n_tasks = len(concepts)
    task_ks = [n_tasks - 1] if args.final_only else list(range(n_tasks))
    if args.only_tasks:
        keep = {int(x) for x in args.only_tasks.split(",")}
        task_ks = [k for k in task_ks if k in keep]
    only = set(args.only_concepts.split(",")) if args.only_concepts else None

    for k in task_ks:
        ckpt = os.path.join(args.ckpt_dir, f"hyper_after_task{k:02d}.pt")
        if not os.path.exists(ckpt):
            print(f"[gen] MISSING {ckpt}, skipping task {k}", flush=True)
            continue
        load_hyper(manager, ckpt, map_location=str(device))
        for j in range(k + 1):
            c = concepts[j]
            if only is not None and c["concept_id"] not in only:
                continue
            ident, cls, cat = c.get("identifier", f"V{j + 1}"), c["class_word"], c["category"]
            gen_repl = " ".join(x for x in (ident, cls) if x)
            clipt_repl = cls
            out_dir = os.path.join(args.out_root, f"after_task{k:02d}", f"task{j:02d}_{c['concept_id']}")
            nimg = gen_concept(bundle, manager, gen_repl, clipt_repl, cat, out_dir,
                               args.num_samples, args.steps, args.guidance_scale, args.seed,
                               task_idx=j,
                               mask_phrase=(" ".join(x for x in (ident, cls) if x)
                                            if cfg.get("token_mask_lora") else None),
                               sample_batch=int(args.sample_batch))
            print(f"[gen] after_task{k:02d} / {c['concept_id']} ({cat}, '{gen_repl}'): {nimg} imgs",
                  flush=True)
    print(f"[gen] DONE -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
