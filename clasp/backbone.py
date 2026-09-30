"""Frozen SD-1.5 or SDXL-base-1.0 backbone loaded through diffusers.

SD-1.5: the UNet's cross-attention context is the last hidden state of the CLIP ViT-L/14 text
encoder, pooled width 768. SDXL: the penultimate hidden states of both text encoders are
concatenated, and the pooled projection of the second (width 1280) is the pooled embedding.
With task keys (`task_cond.key_dim`) the hypernetwork does not read the prompt at all; the
pooled embedding is used only for SDXL's micro-conditioning.
"""

from __future__ import annotations

import dataclasses

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

DEFAULT_SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"

_DTYPES = {"fp32": torch.float32, "float32": torch.float32,
           "fp16": torch.float16, "float16": torch.float16,
           "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def resolve_dtype(name) -> torch.dtype:
    if name is None:
        return torch.float32
    if isinstance(name, torch.dtype):
        return name
    return _DTYPES[str(name).lower()]


@dataclasses.dataclass
class ModelBundle:
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler              # training: add_noise
    ddim_scheduler: DDIMScheduler               # placement probe
    dpm_scheduler: DPMSolverMultistepScheduler  # evaluation matrix
    device: torch.device
    dtype: torch.dtype
    cross_attention_dim: int
    clip_hidden_size: int                       # pooled width (768 / 1280)
    num_train_timesteps: int
    vae_scale_factor: float
    model_id: str
    text_encoder_2: object | None = None
    tokenizer_2: object | None = None
    is_sdxl: bool = False
    default_resolution: int = 512

    @property
    def latent_channels(self) -> int:
        return self.unet.config.in_channels

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (context [B, 77, D], pooled [B, P], attention_mask [B, 77])."""
        batch = self.tokenizer(prompts, padding="max_length",
                               max_length=self.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        if not self.is_sdxl:
            out = self.text_encoder(input_ids=input_ids)
            return out.last_hidden_state, out.pooler_output, attention_mask
        ids2 = self.tokenizer_2(prompts, padding="max_length",
                                max_length=self.tokenizer_2.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to(self.device)
        o1 = self.text_encoder(input_ids=input_ids, output_hidden_states=True)
        o2 = self.text_encoder_2(input_ids=ids2, output_hidden_states=True)
        h1 = o1.hidden_states[-2]
        h2 = o2.hidden_states[-2]
        pooled = o2.text_embeds
        return torch.cat([h1, h2], dim=-1), pooled, attention_mask

    def added_cond(self, batch_size: int, height: int | None = None,
                   width: int | None = None, pooled: torch.Tensor | None = None,
                   orig_size: torch.Tensor | None = None,
                   crop: torch.Tensor | None = None) -> dict:
        """SDXL micro-conditioning {text_embeds, time_ids}; empty on SD-1.5.
        time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)."""
        if not self.is_sdxl:
            return {}
        h = height or self.default_resolution
        w = width or self.default_resolution
        if orig_size is None:
            tid = torch.tensor([h, w, 0, 0, h, w], device=self.device, dtype=self.dtype)
            tid = tid[None].expand(batch_size, -1)
        else:
            o = orig_size.to(self.device, self.dtype)
            c = (crop if crop is not None else torch.zeros_like(orig_size)).to(self.device, self.dtype)
            tgt = torch.tensor([[h, w]], device=self.device, dtype=self.dtype).expand(o.shape[0], -1)
            tid = torch.cat([o, c, tgt], dim=-1)
        pe = pooled.to(self.device, self.dtype)
        if pe.shape[0] != batch_size:
            pe = pe[:1].expand(batch_size, -1)
        return {"text_embeds": pe, "time_ids": tid}

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """images in [-1, 1] [B, 3, H, W] -> scaled latents [B, 4, H/8, W/8]."""
        images = images.to(self.device, dtype=self.vae.dtype)
        posterior = self.vae.encode(images).latent_dist
        z = posterior.sample() * self.vae_scale_factor
        return z.to(self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """latents -> images in [0, 1]; decoded in chunks to bound VAE memory at 1024 px."""
        latents = latents.to(self.vae.dtype) / self.vae_scale_factor
        chunk = 2 if latents.shape[-1] >= 128 else 8
        outs = []
        for i in range(0, latents.shape[0], chunk):
            outs.append(self.vae.decode(latents[i:i + chunk]).sample)
        images = torch.cat(outs)
        return (images / 2 + 0.5).clamp(0, 1)


def load_sd(model_id: str = DEFAULT_SD15, device="cuda", dtype=torch.float32) -> ModelBundle:
    device = torch.device(device if torch.cuda.is_available() or "cpu" not in str(device) else "cpu")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    dtype = resolve_dtype(dtype)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype,
                                                   safety_checker=None,
                                                   requires_safety_checker=False)
    unet, vae = pipe.unet, pipe.vae
    text_encoder, tokenizer = pipe.text_encoder, pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    for module in (unet, vae, text_encoder):
        module.requires_grad_(False)
        module.eval()
        module.to(device)
    del pipe
    return ModelBundle(
        unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        noise_scheduler=noise_scheduler, ddim_scheduler=ddim_scheduler,
        dpm_scheduler=dpm_scheduler, device=device, dtype=dtype,
        cross_attention_dim=int(unet.config.cross_attention_dim),
        clip_hidden_size=int(text_encoder.config.hidden_size),
        num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
        vae_scale_factor=float(vae.config.scaling_factor),
        model_id=model_id)


def load_sdxl(model_id: str = DEFAULT_SDXL, device="cuda", dtype=torch.bfloat16) -> ModelBundle:
    """SDXL-base; the VAE is kept in fp32, the UNet and text encoders in `dtype`."""
    from diffusers import StableDiffusionXLPipeline
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
    unet, vae = pipe.unet, pipe.vae
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    vae = vae.to(dtype=torch.float32)
    for m in (unet, vae, te1, te2):
        m.requires_grad_(False)
        m.eval()
        m.to(device)
    del pipe
    return ModelBundle(
        unet=unet, vae=vae, text_encoder=te1, tokenizer=tok1,
        noise_scheduler=noise_scheduler, ddim_scheduler=ddim_scheduler,
        dpm_scheduler=dpm_scheduler, device=device, dtype=dtype,
        cross_attention_dim=int(unet.config.cross_attention_dim),
        clip_hidden_size=int(te2.config.projection_dim),
        num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
        vae_scale_factor=float(vae.config.scaling_factor),
        model_id=model_id,
        text_encoder_2=te2, tokenizer_2=tok2, is_sdxl=True, default_resolution=1024)


def load_backbone(cfg: dict, device, dtype=None) -> ModelBundle:
    """Backbone named by `sd_model_id`; `dtype` overrides `weight_dtype`."""
    mid = cfg.get("sd_model_id", "")
    dt = dtype or cfg.get("weight_dtype", "fp32")
    if "xl" in str(mid).lower():
        return load_sdxl(model_id=mid, device=device, dtype=dt)
    return load_sd(model_id=mid, device=device, dtype=dt) if mid else load_sd(device=device, dtype=dt)
