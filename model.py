from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig

MASK_ID = 0

def alpha_sigma_from_logsnr(lam: Tensor) -> Tuple[Tensor, Tensor]:
    """Variance-preserving: α=sqrt(sigmoid(λ)), σ=sqrt(sigmoid(-λ))."""
    return torch.sqrt(torch.sigmoid(lam)), torch.sqrt(torch.sigmoid(-lam))

def sigmoid_weight(lam: Tensor, b: float = 0.0) -> Tensor:
    return torch.sigmoid(lam - b)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        emb = t[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class TextVAE(nn.Module):
    """
    first `n_encoder_layers` of Qwen3-2B (frozen by default) + trainable projection head → latent_dim.
    """
    def __init__(
        self,
        model_name: str  = "Qwen/Qwen3-2B",
        latent_dim: int  = 256,
        n_encoder_layers: int  = 8,
        freeze_backbone: bool = True,
        vocab_size: int  = 152001,   # Qwen3 vocab + 1 for MASK
        seq_len: int  = 128,
        dec_d_model: int  = 512,
        dec_n_layers: int  = 2,
    ):
        super().__init__()

        print(f"  Loading Qwen3 encoder ({model_name}, first {n_encoder_layers} layers) …")
        full = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, device_map  = "cpu")

        self.embed_tokens = full.model.embed_tokens
        self.layers = nn.ModuleList(full.model.layers[:n_encoder_layers])
        hidden_size = full.model.embed_tokens.embedding_dim
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        del full

        if freeze_backbone:
            for p in self.embed_tokens.parameters():
                p.requires_grad = False
            for p in self.layers.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, latent_dim),
        )

        self.ce_decoder = _LightDecoder(
            vocab_size = vocab_size,
            seq_len    = seq_len,
            d_model    = dec_d_model,
            n_layers   = dec_n_layers,
            latent_dim = latent_dim,
        )

    def encode(self, tokens: Tensor) -> Tensor:
        x = self.embed_tokens(tokens).to(self.proj[0].weight.dtype)
        for layer in self.layers:
            x = layer(x)[0]
        x = self.norm(x).mean(dim=1)
        return self.proj(x)

    def decode_logits(self, tokens: Tensor, z: Tensor) -> Tensor:
        return self.ce_decoder(tokens, z)

    def reconstruction_loss(self, tokens: Tensor) -> Tensor:
        z = self.encode(tokens)
        logits = self.decode_logits(tokens, z)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))


class _LightDecoder(nn.Module):
    """Small transformer CE decoder used only for stage-1 warm-up."""

    def __init__(self, vocab_size, seq_len, d_model, n_layers, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.latent_in = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor, z: Tensor) -> Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos(pos)
        cls = self.latent_in(z).unsqueeze(1)
        x = torch.cat([cls, x], dim=1)
        return self.out(self.transformer(x)[:, 1:, :])

class LatentDiffusionPrior(nn.Module):
    """
    MLP-transformer denoising z_t → z_clean in latent space.
    Unweighted ELBO loss (w=1) required for the bitrate bound.
    Also exposes sigmoid_loss() for stage-2 base model training.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        d_model: int = 512,
        n_layers: int = 4,
        lam_min_fixed: float = 5.0,
        lam_max: float = 10.0,
    ):
        super().__init__()
        self.lam_min_fixed = lam_min_fixed
        self.lam_max_train = lam_max

        self.time_emb = SinusoidalPosEmb(d_model)
        self.in_proj = nn.Linear(latent_dim, d_model)
        layers: List[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(d_model, latent_dim)

    def _logsnr_schedule(self, t: Tensor) -> Tensor:
        return self.lam_min_fixed + (1 - t) * (self.lam_max_train - self.lam_min_fixed)

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        return self.out(self.net(self.in_proj(z_t) + self.time_emb(t)))

    def loss(self, z_clean: Tensor) -> Tensor:
        """Unweighted ELBO — required for bitrate bound."""
        B = z_clean.size(0)
        t = torch.rand(B, device=z_clean.device)
        lam = self._logsnr_schedule(t)
        a, s = alpha_sigma_from_logsnr(lam)
        z_t = a[:, None] * z_clean + s[:, None] * torch.randn_like(z_clean)
        z_hat = self.forward(z_t, t)
        dlam = torch.full_like(lam, self.lam_max_train - self.lam_min_fixed)
        return (dlam * torch.exp(lam / 2) * ((z_clean - z_hat) ** 2).mean(-1)).mean()

    def sigmoid_loss(self, z_clean: Tensor) -> Tensor:
        """Sigmoid-weighted loss for stage-2 base model."""
        B = z_clean.size(0)
        t = torch.rand(B, device=z_clean.device)
        lam = self._logsnr_schedule(t)
        a, s = alpha_sigma_from_logsnr(lam)
        z_t = a[:, None] * z_clean + s[:, None] * torch.randn_like(z_clean)
        z_hat = self.forward(z_t, t)
        dlam = torch.full_like(lam, self.lam_max_train - self.lam_min_fixed)
        w = sigmoid_weight(lam)
        return (w * dlam * torch.exp(lam / 2) * ((z_clean - z_hat) ** 2).mean(-1)).mean()

    @torch.no_grad()
    def sample_z0(self, z1: Tensor, n_steps: int = 50) -> Tensor:
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=z1.device)
        z  = z1
        for i in range(n_steps):
            t_now = ts[i].expand(z.size(0))
            t_next = ts[i + 1].expand(z.size(0))
            an, sn = alpha_sigma_from_logsnr(self._logsnr_schedule(t_now))
            ax, sx = alpha_sigma_from_logsnr(self._logsnr_schedule(t_next))
            z_hat = self.forward(z, t_now)
            mu = (ax[:, None] * sn[:, None] ** 2 * z + an[:, None] * sx[:, None] ** 2 * z_hat) \
                     / (an[:, None] * sn[:, None] ** 2 + 1e-8)
            noise = (sx[:, None] * sn[:, None] / (sn[:, None] + 1e-8)).clamp(min=1e-3)
            z = mu + noise * torch.randn_like(z)
        return z

class MaskedTextDecoder(nn.Module):
    """
    Absorbing-diffusion decoder p_θ(x | z₀) built on the full Qwen3-2B model.

    z₀ conditioning is injected by prepending `n_latent_tokens` soft tokens
    derived from z₀ via a small MLP.  The backbone is fine-tuned end-to-end
    by default (freeze_backbone=False); set True to train only the projection.

    The MaskGIT-style confidence-based sampler commits the most-confident
    tokens each step, preventing early lock-in of bad predictions.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-2B",
        latent_dim: int = 256,
        n_latent_tokens: int = 8,
        sigmoid_bias: float = 0.0,
        loss_factor: float = 1.5,
        freeze_backbone: bool = False,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()

        self.sigmoid_bias = sigmoid_bias
        self.loss_factor = loss_factor
        self.n_latent_tokens = n_latent_tokens

        print(f"  Loading Qwen3 decoder backbone ({model_name}) …")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            device_map  = "cpu",
        )
        hidden_size = self.backbone.config.hidden_size
        self.vocab_size = vocab_size or self.backbone.config.vocab_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.time_emb = SinusoidalPosEmb(hidden_size)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_latent_tokens * hidden_size),
        )
        self.time_proj = nn.Linear(hidden_size, hidden_size)

    def _prefix_embeds(self, z0: Tensor, t: Tensor) -> Tensor:
        """Build (B, n_latent_tokens, H) prefix from z₀ and t."""
        B = z0.size(0)
        dtype = self.backbone.model.embed_tokens.weight.dtype
        prefix = self.latent_proj(z0.to(dtype)).reshape(B, self.n_latent_tokens, -1)
        t_bias = self.time_proj(self.time_emb(t).to(dtype)).unsqueeze(1)
        return prefix + t_bias

    def forward(self, x_t: Tensor, z0: Tensor, t: Tensor) -> Tensor:
        """x_t: (B, L) masked tokens → logits: (B, L, vocab_size)"""
        prefix = self._prefix_embeds(z0, t)
        tok_emb = self.backbone.model.embed_tokens(x_t)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False)
        return out.logits[:, self.n_latent_tokens:, :self.vocab_size]

    def loss(self, tokens: Tensor, z0: Tensor) -> Tensor:
        B = tokens.size(0)
        t = torch.rand(B, device=tokens.device) * 0.95 + 0.025
        mask = torch.bernoulli(t[:, None].expand_as(tokens)).bool()
        x_t = tokens.clone()
        x_t[mask] = MASK_ID

        logits = self.forward(x_t, z0, t)
        ce = F.cross_entropy(logits[mask], tokens[mask], reduction="mean") \
                 if mask.any() else logits.new_zeros(())

        lam_t = torch.log((1 - t + 1e-6) / (t + 1e-6))
        w = self.loss_factor * sigmoid_weight(lam_t, self.sigmoid_bias)
        return w.mean() * ce

    @torch.no_grad()
    def sample(self, z0: Tensor, seq_len: int, n_steps: int = 20) -> Tensor:
        """MaskGIT-style confidence-based iterative unmasking."""
        B = z0.size(0)
        x = torch.full((B, seq_len), MASK_ID, dtype=torch.long, device=z0.device)

        for i in range(n_steps, 0, -1):
            t = torch.full((B,), i / n_steps, device=z0.device)
            logits = self.forward(x, z0, t)
            probs = torch.softmax(logits, dim=-1)

            sampled = torch.multinomial(probs.reshape(-1, self.vocab_size), 1).reshape(B, seq_len)
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            n_unmask = max(1, round(seq_len * (1 - (i - 1) / n_steps)))
            still_masked = x == MASK_ID
            confidence[~still_masked] = -1.0

            topk = confidence.topk(min(n_unmask, int(still_masked.sum(dim=-1).max().item())), dim=-1).indices
            commit = torch.zeros_like(still_masked).scatter_(-1, topk, True) & still_masked
            x[commit] = sampled[commit]

        # Fill any remaining masks with argmax
        if (x == MASK_ID).any():
            t = torch.full((B,), 1 / n_steps, device=z0.device)
            logits = self.forward(x, z0, t)
            x[x == MASK_ID] = logits.argmax(-1)[x == MASK_ID]

        return x