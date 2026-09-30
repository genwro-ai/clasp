#!/bin/bash
# Commands behind the results. Each block is independent; run the ones you need.
# One GPU. SD-1.5 training of ten concepts takes about an hour on a GH200, SDXL 5.5-7 h.
set -euo pipefail

# --- data preparation (once) ------------------------------------------------------------
uv run python -m scripts.download_data
uv run python -m scripts.prepare_cc101 --src data/benchmark_dataset
uv run --extra segment python -m scripts.segment_subjects --config configs/sd15_bench10.yaml --out data/seg
uv run python -m scripts.segment_gsam --config configs/sd15_seq100.yaml --out data/seg --only_prefix cc101_
uv run python -m scripts.segment_overrides --config configs/sd15_seq100.yaml --out data/seg
uv run python -m scripts.make_backgrounds --out data/backgrounds
uv run python -m scripts.make_backgrounds --out data/backgrounds_sdxl --sdxl --size 1024

# --- ten concepts, SD-1.5 (LoRA-scale sweep, 55-cell forgetting) -------------------------
uv run python -m clasp.train --config configs/sd15_bench10.yaml
for S in 0.30 0.45 0.60 0.75 0.90 1.05; do
  OUT=outputs/sd15_bench10/matrix/s${S/./}
  uv run python -m clasp.generate --config configs/sd15_bench10.yaml \
      --ckpt_dir outputs/sd15_bench10/ckpts --out_root "$OUT" \
      --num_samples 10 --lora_scale "$S" --sample_batch 10 --eval_dtype fp16
  uv run python -m clasp.metrics --config configs/sd15_bench10.yaml --eval_root "$OUT"
done
# seeds 2025 and 2026: copy the config and change `seed` and `output_dir`

# --- placement, model after task 10 ------------------------------------------------------
CK=outputs/sd15_bench10/ckpts/hyper_after_task09.pt
uv run python -m clasp.placement --config configs/sd15_bench10.yaml --ckpt "$CK" --grid 0:0,1:1,2:0.3 \
    --n 3 --scale 0.7 --steps 50 --scene "on a beach"
uv run python -m clasp.placement --config configs/sd15_bench10.yaml --ckpt "$CK" --grid 2:0.3 \
    --bootstrap 15 --scaffold_steps 10 --n 3 --scale 0.7 --steps 50 --scene "on a beach"
uv run python -m clasp.placement --config configs/sd15_bench10.yaml --ckpt "$CK" --grid 2:0.3 \
    --layout prompt --n 3 --scale 0.7 --steps 50 --scene "on a beach"
uv run python -m clasp.placement --config configs/sd15_bench10.yaml --ckpt "$CK" --grid 2:0.3 \
    --layout regional --n 3 --scale 0.7 --steps 50 --scene "on a beach"

# --- fifty concepts, SD-1.5 (retention after tasks 10 to 50) -----------------------------
# the first ten tasks are the ten-concept run; the sequence continues from its checkpoint
uv run python -m clasp.train --config configs/sd15_seq50.yaml --end_task 9
uv run python -m clasp.train --config configs/sd15_seq50.yaml \
    --init_ckpt outputs/sd15_seq50/ckpts/hyper_after_task09.pt --start_task 10
TEN=cifc_dog,cifc_duck_toy,cifc_cat,cifc_backpack,cifc_teddybear,cifc_painting,cifc_dog2,cifc_drawing,cifc_cat2,cifc_ink_painting
OUT=outputs/sd15_seq50/retention/s045
uv run python -m clasp.generate --config configs/sd15_seq50.yaml \
    --ckpt_dir outputs/sd15_seq50/ckpts --out_root "$OUT" \
    --num_samples 10 --lora_scale 0.45 --sample_batch 10 --eval_dtype fp16 \
    --only_tasks 9,19,29,36,49 --only_concepts "$TEN"
uv run python -m clasp.metrics --config configs/sd15_seq50.yaml --eval_root "$OUT"

# --- a hundred concepts, SD-1.5 (retention and placement at T = 100) ----------------------
# continues from the fifty-concept checkpoint; load_hyper widens it to the hundred slots
uv run python -m clasp.train --config configs/sd15_seq100.yaml \
    --init_ckpt outputs/sd15_seq50/ckpts/hyper_after_task49.pt --start_task 50
OUT=outputs/sd15_seq100/retention/s045
uv run python -m clasp.generate --config configs/sd15_seq100.yaml \
    --ckpt_dir outputs/sd15_seq100/ckpts --out_root "$OUT" \
    --num_samples 10 --lora_scale 0.45 --sample_batch 10 --eval_dtype fp16 \
    --only_tasks 99 --only_concepts "$TEN"
uv run python -m clasp.metrics --config configs/sd15_seq100.yaml --eval_root "$OUT"
uv run python -m clasp.placement --config configs/sd15_seq100.yaml \
    --ckpt outputs/sd15_seq100/ckpts/hyper_after_task99.pt --grid 2:0.3 \
    --bootstrap 15 --scaffold_steps 10 --n 3 --scale 0.7 --steps 50 --scene "on a beach" \
    --only_concepts cifc_dog,cifc_duck_toy,cifc_cat,cifc_backpack,cifc_teddybear,cifc_dog2,cifc_cat2

# --- SDXL, second backbone ---------------------------------------------------------------
uv run python -m clasp.train --config configs/sdxl_bench10.yaml
for S in 0.20 0.25 0.30 0.40 0.50; do
  OUT=outputs/sdxl_bench10/final/s${S/./}
  uv run python -m clasp.generate --config configs/sdxl_bench10.yaml \
      --ckpt_dir outputs/sdxl_bench10/ckpts --out_root "$OUT" --final_only \
      --num_samples 10 --lora_scale "$S" --sample_batch 10 --eval_dtype fp16
  uv run python -m clasp.metrics --config configs/sdxl_bench10.yaml --eval_root "$OUT"
done
# separate scale for the output projection (held at 0.50, the reading projections swept):
#   --lora_scale_map "attn2.to_out.0=0.50,attn2.to_q=S,attn2.to_k=S,attn2.to_v=S"
