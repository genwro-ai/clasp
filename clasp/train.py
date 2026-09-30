"""Sequential training of the hypernetwork over a sequence of concepts.

For every task t the hypernetwork minimizes the denoising loss on concept t plus the output
regularizer on the adapters it generates for the embeddings of tasks 1..t-1:

  1. trial step on the new concept: delta = -eta * sg[grad L_rec]
  2. L = L_rec(phi) + beta * mean_{i<t} ||dW_{phi+delta}(v_i) - dW_{phi*_t}(v_i)||_F^2 / (d_in d_out)

with phi*_t the parameters at the start of the task. The localization module (token head, FiLM
head, read projections, gates) is trained by the fit term only. On half of the steps of an object
concept, a segmented subject is pasted at a random position and scale onto a background generated
by the frozen backbone, the box is passed to the branch, and the timestep is drawn from the
high-noise half. A checkpoint is written after every task.

Run:
  python -m clasp.train --config configs/sd15_bench10.yaml
  # the fifty-concept sequence continues from the ten-concept checkpoint:
  python -m clasp.train --config configs/sd15_seq50.yaml --end_task 9
  python -m clasp.train --config configs/sd15_seq50.yaml \
      --init_ckpt outputs/sd15_seq50/ckpts/hyper_after_task09.pt --start_task 10
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .backbone import load_backbone
from .common import load_config, load_hyper, save_hyper, set_seed
from .data import ConceptDataset, collate_fn, specs_from_config, token_span_mask
from .hypernet import build_hyper
from .injection import DEFAULT_TARGETS
from .localization import set_grounded
from .regularizer import reg_dw

PASTE_SCALE = (0.45, 0.85)   # pasted subject's long side as a fraction of the frame


def parse_args():
    p = argparse.ArgumentParser(description="CLASP sequential training")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--init_ckpt", default=None,
                   help="checkpoint to resume from, e.g. ckpts/hyper_after_task09.pt")
    p.add_argument("--start_task", type=int, default=0,
                   help="first task to train; needs --init_ckpt holding the state after "
                        "task start_task-1")
    p.add_argument("--end_task", type=int, default=None,
                   help="last task to train (inclusive)")
    return p.parse_args()


def _load_banks(specs, seg_dir, bg_dir, res):
    """Backgrounds [3, res, res] in [-1, 1] and per-concept RGBA cutouts (rgb, alpha)."""
    bg_bank, seg_bank = [], {}
    for f in sorted(glob.glob(os.path.join(bg_dir, "*"))):
        im = Image.open(f).convert("RGB").resize((res, res), Image.LANCZOS)
        bg_bank.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 127.5 - 1.0)
    for spec in specs:
        cuts = []
        for f in sorted(glob.glob(os.path.join(seg_dir, spec.concept_id, "*.png"))):
            im = Image.open(f).convert("RGBA")
            t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()
            cuts.append((t[:3] / 127.5 - 1.0, t[3:4] / 255.0))
        if cuts:
            seg_bank[spec.concept_id] = cuts
    return bg_bank, seg_bank


def _paste(images, cuts, bg_bank, alpha_erode):
    """Composite one cutout per sample onto a random background. -> (images, boxes)."""
    H = images.shape[-1]
    comp, boxes = [], []
    for _ in range(images.shape[0]):
        rgb, al = cuts[int(torch.randint(0, len(cuts), (1,)).item())]
        ch, cw = rgb.shape[-2:]
        sc = float(torch.empty(1).uniform_(PASTE_SCALE[0], PASTE_SCALE[1]).item())
        r = sc * H / max(ch, cw)
        nh, nw = max(8, int(ch * r)), max(8, int(cw * r))
        rgb = F.interpolate(rgb[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
        al = F.interpolate(al[None], size=(nh, nw), mode="bilinear",
                           align_corners=False)[0].clamp(0, 1)
        if alpha_erode > 0:
            # shrink the matte so the source background does not leave a rim around the subject
            ksz = 2 * alpha_erode + 1
            al = -F.max_pool2d(-al[None], ksz, stride=1, padding=alpha_erode)[0]
        x0 = int(torch.randint(0, H - nw + 1, (1,)).item())
        y0 = int(torch.randint(0, H - nh + 1, (1,)).item())
        boxes.append(((x0 + nw / 2) / H, (y0 + nh / 2) / H, nw / H, nh / H))
        bg = bg_bank[int(torch.randint(0, len(bg_bank), (1,)).item())].clone()
        reg = bg[:, y0:y0 + nh, x0:x0 + nw]
        bg[:, y0:y0 + nh, x0:x0 + nw] = reg * (1 - al) + rgb * al
        comp.append(bg)
    return torch.stack(comp), boxes


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 2024)))

    resolution = int(cfg.get("resolution", 512))
    output_dir = args.output_dir or cfg.get("output_dir", "./outputs/run")
    train = cfg.get("training", {})
    steps_per_task = int(train.get("steps_per_task", 800))
    batch_size = int(train.get("batch_size", 2))
    lr = float(train.get("lr", 1e-4))
    grad_clip = float(train.get("grad_clip", 1.0))
    log_every = int(train.get("log_every", 50))
    reg_cfg = cfg.get("reg", {})
    reg_weight = float(reg_cfg.get("weight", 0.0))
    lookahead_lr = float(reg_cfg.get("lookahead_lr", lr))
    box_aug_p = float(train.get("box_aug_p", 0.5))
    box_t_min_frac = float(train.get("box_t_min_frac", 0.0))
    alpha_erode = int(train.get("alpha_erode", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_backbone(cfg, device)
    manager = build_hyper(bundle, target_modules=tuple(cfg.get("target_modules", DEFAULT_TARGETS)),
                          n_tasks=len(cfg["concepts"]), task_cond=cfg.get("task_cond"),
                          **cfg.get("hyper", {}))
    manager.train()
    specs = specs_from_config(cfg["concepts"])
    n_tasks = len(specs)

    bg_bank, seg_bank = _load_banks(specs, train["seg_dir"], train["bg_dir"], resolution)
    print(f"[train] backgrounds {len(bg_bank)}, cutouts "
          f"{ {k: len(v) for k, v in seg_bank.items()} }", flush=True)
    if manager.ground_cond:
        print(f"[train] grounded attention on {set_grounded(bundle.unet, manager)} attn2 layers",
              flush=True)
    tm_enabled = bool(cfg.get("token_mask_lora", False))
    print(f"[train] {n_tasks} tasks | beta={reg_weight} lookahead_lr={lookahead_lr} | "
          f"{steps_per_task} steps/task | bs={batch_size} | lr={lr}", flush=True)
    unet = bundle.unet

    named = list(manager.heads.named_parameters())       # content heads, stable across tasks
    params = [p for _, p in named]
    loc_params = manager.localization_parameters()
    learned_conds = []  # conditioning of every learned concept, for the regularizer

    if args.start_task > 0:
        if not args.init_ckpt:
            raise SystemExit("--start_task needs --init_ckpt")
        load_hyper(manager, args.init_ckpt, map_location=str(device))
        got = int(manager.basis_count.item())
        if got != args.start_task:
            raise SystemExit(f"checkpoint has basis_count={got}, but --start_task={args.start_task}")
        # these are not stored in the checkpoint; rebuild them as the loop would have
        with torch.no_grad():
            for j in range(args.start_task):
                _, pooled_j, _ = bundle.encode_text([specs[j].diag_prompt])
                learned_conds.append(manager.condition(pooled_j, j)[0].detach())
        print(f"[train] resumed from {args.init_ckpt} at task {args.start_task}", flush=True)

    for k, spec in enumerate(specs):
        if k < args.start_task:
            continue
        if args.end_task is not None and k > args.end_task:
            print(f"[train] stopped after task {args.end_task}", flush=True)
            raise SystemExit(0)
        # weights persist across tasks, the optimizer is fresh per task
        optimizer = torch.optim.AdamW([{"params": manager.hyper_parameters()}], lr=lr,
                                      weight_decay=float(train.get("weight_decay", 0.0)))
        with torch.no_grad():
            manager.set_canonical(k)
        loader = DataLoader(ConceptDataset(spec, resolution, augment=bool(train.get("augment", False))),
                            batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=collate_fn, num_workers=int(train.get("num_workers", 2)))
        data_iter = itertools.cycle(loader)
        _ds = loader.dataset
        _caps = sorted({_ds._caption(os.path.splitext(os.path.basename(q))[0]) for q in _ds.paths})
        print(f"[train] captions {spec.concept_id}: " + " | ".join(_caps), flush=True)

        targets, learned = None, None
        if reg_weight > 0 and learned_conds:
            learned = torch.stack(learned_conds, 0)
            with torch.no_grad():
                lora = manager.generate_lora(learned)            # snapshot phi*_t
                targets = {n: (a.detach(), b.detach()) for n, (a, b) in lora.items()}

        for step in range(steps_per_task):
            batch = next(data_iter)
            images = batch["pixel_values"].to(device)
            captions = batch["captions"]
            bsz = images.shape[0]
            orig_size, crop = batch["orig_size"], batch["crop"]

            if manager.ground_cond:
                manager.cond_box = None
                cuts = seg_bank.get(spec.concept_id)
                if cuts:
                    if torch.rand(1).item() < box_aug_p:
                        comp, boxes = _paste(images, cuts, bg_bank, alpha_erode)
                        images = comp.to(device)
                        H = images.shape[-1]
                        orig_size = torch.full((bsz, 2), H)
                        crop = torch.zeros(bsz, 2, dtype=torch.long)
                        manager.cond_box = boxes[0] if len(boxes) == 1 else boxes
                else:
                    # concepts without cutouts (styles) draw one number per step and never
                    # paste; kept so the random draws match the runs behind our results
                    torch.rand(1).item()
            cond_hidden, pooled, _ = bundle.encode_text(captions)
            z0 = bundle.encode_images(images)
            noise = torch.randn_like(z0)
            t = torch.randint(0, bundle.num_train_timesteps, (bsz,), device=device)
            if manager.cond_box is not None and box_t_min_frac > 0:
                # pasted steps use the high-noise half, where z_t carries little layout
                lo = int(box_t_min_frac * bundle.num_train_timesteps)
                t = torch.randint(lo, bundle.num_train_timesteps, (bsz,), device=device)
            z_t = bundle.noise_scheduler.add_noise(z0, noise, t)

            tok_mask = (token_span_mask(bundle.tokenizer, captions, spec.replacement).to(device)
                        if tm_enabled else None)
            if manager.ground_cond:
                manager.set_ground(k, manager.cond_box)
            manager.set_context(pooled, task_idx=k, token_mask=tok_mask)
            manager.compute_and_cache_loras()
            manager.enable_lora()
            ac = bundle.added_cond(z_t.shape[0], resolution, resolution, pooled=pooled,
                                   orig_size=orig_size, crop=crop) if bundle.is_sdxl else None
            eps_pred = unet(z_t, t, encoder_hidden_states=cond_hidden, added_cond_kwargs=ac).sample
            loss = F.mse_loss(eps_pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            reg_val = 0.0
            if targets is not None:
                # stage 1: trial step on the new concept only (detached)
                g_all = torch.autograd.grad(loss, params + loc_params,
                                            retain_graph=False, allow_unused=True)
                g = g_all[:len(params)]
                g_loc = g_all[len(g_all) - len(loc_params):] if loc_params else []
                delta = {nm: (-lookahead_lr * gi).detach() for (nm, _), gi in zip(named, g)}
                # stage 2: regularize the outputs at the lookahead parameters
                perturbed = {nm: p + delta[nm] for nm, p in named}
                reg = reg_weight * reg_dw(manager.lora_from_params(learned, perturbed), targets)
                g_reg = torch.autograd.grad(reg, params)
                for p, gi, gr in zip(params, g, g_reg):
                    p.grad = gi + gr
                for p, gi in zip(loc_params, g_loc):
                    if gi is None:
                        continue
                    p.grad = gi
                reg_val = float(reg.item())
            else:
                loss.backward()
            gnorm = 0.0
            if grad_clip > 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(params + loc_params, grad_clip))
            optimizer.step()

            if step % log_every == 0 or step == steps_per_task - 1:
                print(f"[train] task {k}:{spec.concept_id} | step {step:4d} | loss {loss.item():.4f}"
                      f" | gnorm {gnorm:.3f}" + (f" | reg {reg_val:.4f}" if targets is not None else ""),
                      flush=True)

        # freeze this task's direction before the checkpoint, then record its conditioning
        with torch.no_grad():
            manager.freeze_task_basis(k)
            if reg_weight > 0:
                _, pooled_k, _ = bundle.encode_text([spec.diag_prompt])
                learned_conds.append(manager.condition(pooled_k, k)[0].detach())
        save_hyper(manager, os.path.join(output_dir, "ckpts", f"hyper_after_task{k:02d}.pt"))
        print(f"[train] done task {k}:{spec.concept_id}", flush=True)

    save_hyper(manager, os.path.join(output_dir, "hyper.pt"))
    print(f"[train] DONE -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
