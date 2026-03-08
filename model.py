from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

def log_snr_cosine(t: Tensor, lam_min: float = -10.0, lam_max: float = 10.0) -> Tensor:
    """Cosine log-SNR schedule mapped to [lam_min, lam_max]."""
    angle = math.pi / 2 * t
    snr = torch.cos(angle) ** 2 / (torch.sin(angle) ** 2 + 1e-8)
    lam = torch.log(snr.clamp(min=1e-8))
    # rescale linearly to desired range
    lam = lam_min + (lam - lam_min) * (lam_max - lam_min) / (lam.max() - lam.min() + 1e-8)
    return lam


def alpha_sigma_from_logsnr(lam: Tensor) -> Tuple[Tensor, Tensor]:
    """Variance-preserving parameterisation: α² = sigmoid(λ), σ² = sigmoid(−λ)."""
    alpha = torch.sqrt(torch.sigmoid(lam))
    sigma = torch.sqrt(torch.sigmoid(-lam))
    return alpha, sigma


def sigmoid_weight(lam: Tensor, b: float = 0.0) -> Tensor:
    """Reweighting function w(λ) = sigmoid(λ − b) used for decoder loss."""
    return torch.sigmoid(lam - b)

class SinusoidalPosEmb(nn.Module):
    """1-D sinusoidal embedding for a scalar timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) float in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t[:, None] * freqs[None, :]  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)

class TransformerEncoder(nn.Module):
    """Bidirectional transformer encoder → mean-pooled latent."""

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int,
                 latent_dim: int, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, L) long → z_clean: (B, latent_dim)"""
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos(pos)
        x = self.transformer(x)
        x = x.mean(dim=1)          # mean pooling
        return self.proj(x)


class TransformerDecoder(nn.Module):
    """Simple auto-regressive-style decoder conditioned on a latent vector.
    Used only for the stage-1 reconstruction sanity-check loss (not the
    diffusion decoder).
    """

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int,
                 latent_dim: int, n_heads: int = 4):
        super().__init__()
        self.seq_len   = seq_len
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos       = nn.Embedding(seq_len, d_model)
        self.latent_in = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor, z: Tensor) -> Tensor:
        """tokens: (B, L), z: (B, latent_dim) → logits: (B, L, V)"""
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0)
        x    = self.embed(tokens) + self.pos(pos)
        # prepend latent as a "CLS" token
        cls  = self.latent_in(z).unsqueeze(1)          # (B, 1, d_model)
        x    = torch.cat([cls, x], dim=1)              # (B, L+1, d_model)
        x    = self.transformer(x)
        x    = x[:, 1:, :]                             # drop cls output
        return self.out(x)


class TextVAE(nn.Module):
    """Deterministic encoder + CE decoder (no KL — regularisation comes from
    the diffusion prior in UL).
    """

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int,
                 latent_dim: int, n_heads: int = 4):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, seq_len, d_model, n_layers,
                                          latent_dim, n_heads)
        self.ce_decoder = TransformerDecoder(vocab_size, seq_len, d_model, n_layers,
                                             latent_dim, n_heads)

    def encode(self, tokens: Tensor) -> Tensor:
        return self.encoder(tokens)

    def decode_logits(self, tokens: Tensor, z: Tensor) -> Tensor:
        return self.ce_decoder(tokens, z)

    def reconstruction_loss(self, tokens: Tensor) -> Tensor:
        """Cross-entropy reconstruction loss (teacher-forced)."""
        z      = self.encode(tokens)
        logits = self.decode_logits(tokens, z)          # (B, L, V)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tokens.reshape(-1))

class LatentDiffusionPrior(nn.Module):
    """
    Transformer that denoises z_t → z_clean in the continuous latent space.

    The model is conditioned on the noise level t (scalar per sample).
    Training objective: unweighted MSE ELBO (w=1) as required by UL so that
    the prior accurately measures the latent bitrate.

    Architecture: simple MLP-mixer-style transformer operating on a flat
    latent vector (no spatial structure needed for the 1-D latent we use).
    """

    def __init__(self, latent_dim: int, d_model: int, n_layers: int,
                 lam_min: float = -10.0, lam_max_train: float = 10.0,
                 lam_min_fixed: float = 5.0):
        super().__init__()
        self.latent_dim     = latent_dim
        self.lam_min        = lam_min
        self.lam_max_train  = lam_max_train
        # λ(0) = 5  →  σ₀ = sqrt(sigmoid(−5)) ≈ 0.08
        self.lam_min_fixed  = lam_min_fixed

        self.time_emb  = SinusoidalPosEmb(d_model)
        self.in_proj   = nn.Linear(latent_dim, d_model)
        layers: List[nn.Module] = []
        for _ in range(n_layers):
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Linear(d_model, d_model * 4))
            layers.append(nn.GELU())
            layers.append(nn.Linear(d_model * 4, d_model))
        self.net    = nn.Sequential(*layers)
        self.out    = nn.Linear(d_model, latent_dim)

    def _logsnr_schedule(self, t: Tensor) -> Tensor:
        """Map t ∈ [0,1] to log-SNR in [lam_min_fixed, lam_max_train].
        t=0 → λ_min_fixed (least noisy end seen by prior).
        t=1 → very noisy (≈ pure Gaussian).
        """
        # linear in λ space
        lam = self.lam_min_fixed + (1 - t) * (self.lam_max_train - self.lam_min_fixed)
        return lam

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        """Predict z_clean from z_t.
        z_t: (B, latent_dim), t: (B,) ∈ [0, 1]
        Returns z_hat: (B, latent_dim)
        """
        h = self.in_proj(z_t) + self.time_emb(t)
        h = self.net(h)
        return self.out(h)

    def loss(self, z_clean: Tensor) -> Tensor:
        """
        Unweighted diffusion ELBO loss on latents.
        L_prior = E_t[ dλ/dt · exp(λ/2) · ||z_clean − ẑ(z_t, t)||² ]
        We approximate dλ/dt · exp(λ/2) by sampling t uniformly and
        computing an importance-weighted MSE.
        """
        B = z_clean.size(0)
        t = torch.rand(B, device=z_clean.device)
        lam = self._logsnr_schedule(t)           # (B,)
        alpha, sigma = alpha_sigma_from_logsnr(lam)

        eps   = torch.randn_like(z_clean)
        z_t   = alpha[:, None] * z_clean + sigma[:, None] * eps

        z_hat = self.forward(z_t, t)

        # Unweighted SNR loss (w=1 required for bitrate bound)
        # weight = |dλ/dt| · exp(λ/2)
        # For linear schedule: dλ/dt = -(lam_max - lam_min_fixed)
        dlam_dt = torch.full_like(lam, self.lam_max_train - self.lam_min_fixed)
        weight  = dlam_dt * torch.exp(lam / 2)
        mse     = ((z_clean - z_hat) ** 2).mean(dim=-1)   # (B,)
        return (weight * mse).mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_z0(self, z1: Tensor, n_steps: int = 50) -> Tensor:
        """
        Ancestral sampler: z₁ (pure noise) → z₀ (slightly noisy latent).
        Returns z₀ at λ = lam_min_fixed.
        """
        device = z1.device
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        z = z1
        for i in range(n_steps):
            t_now  = ts[i].expand(z.size(0))
            t_next = ts[i + 1].expand(z.size(0))

            lam_now  = self._logsnr_schedule(t_now)
            lam_next = self._logsnr_schedule(t_next)

            a_now,  s_now  = alpha_sigma_from_logsnr(lam_now)
            a_next, s_next = alpha_sigma_from_logsnr(lam_next)

            z_hat = self.forward(z, t_now)

            # DDPM-style reverse step
            coef1  = a_next[:, None] * s_now[:, None] ** 2 / (a_now[:, None] * s_next[:, None] ** 2 + 1e-8)
            coef2  = a_now[:, None] * s_next[:, None] ** 2 / (s_now[:, None] ** 2 + 1e-8)
            mu     = coef1 * z + coef2 * z_hat

            noise_std = (s_next[:, None] * s_now[:, None] / (s_now[:, None] + 1e-8)).clamp(min=0)
            z = mu + noise_std * torch.randn_like(z)

        return z

MASK_ID = 0   # reserve token 0 as [MASK]


class MaskedTextDecoder(nn.Module):
    """
    Masked (absorbing) diffusion decoder: p_θ(x | z₀).

    At each timestep t ∈ [0, 1]:
      - a fraction t of tokens is replaced with [MASK]
      - the model predicts the original token at each masked position
    conditioned on z₀ (the slightly-noisy latent from the prior).

    The decoder loss is the *reweighted* CE:
        L_dec = E_t [ w(λ_x(t)) · CE(x, x̂(x_t, z₀, t)) ]
    where w(λ) = c_lf · sigmoid(λ − b) (sigmoid reweighting).
    """

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int,
                 latent_dim: int, n_heads: int = 4,
                 sigmoid_bias: float = 0.0, loss_factor: float = 1.5):
        super().__init__()
        self.vocab_size   = vocab_size
        self.seq_len      = seq_len
        self.sigmoid_bias = sigmoid_bias
        self.loss_factor  = loss_factor

        self.tok_embed    = nn.Embedding(vocab_size, d_model)
        self.pos_embed    = nn.Embedding(seq_len, d_model)
        self.time_emb     = SinusoidalPosEmb(d_model)
        self.latent_in    = nn.Linear(latent_dim, d_model)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.transformer  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out          = nn.Linear(d_model, vocab_size)

    def _mask_tokens(self, tokens: Tensor, t: Tensor) -> Tensor:
        """Replace each token independently with [MASK] with probability t."""
        mask_prob = t[:, None].expand_as(tokens)        # (B, L)
        mask      = torch.bernoulli(mask_prob).bool()
        masked    = tokens.clone()
        masked[mask] = MASK_ID
        return masked, mask

    def forward(self, x_t: Tensor, z0: Tensor, t: Tensor) -> Tensor:
        """
        x_t : (B, L) - partially masked token sequence
        z0  : (B, latent_dim) - conditioning latent
        t   : (B,) - noise level in [0, 1]
        Returns logits (B, L, vocab_size).
        """
        B, L = x_t.shape
        pos  = torch.arange(L, device=x_t.device).unsqueeze(0)
        temb = self.time_emb(t)                          # (B, d_model)
        lc   = self.latent_in(z0)                        # (B, d_model)
        x    = self.tok_embed(x_t) + self.pos_embed(pos) # (B, L, d_model)
        # inject time & latent conditioning as a prepended token
        cond = (temb + lc).unsqueeze(1)                  # (B, 1, d_model)
        x    = torch.cat([cond, x], dim=1)               # (B, L+1, d_model)
        x    = self.transformer(x)
        x    = x[:, 1:, :]                               # strip cond token
        return self.out(x)                               # (B, L, V)

    def loss(self, tokens: Tensor, z0: Tensor) -> Tensor:
        """
        Reweighted masked-diffusion ELBO loss.

        Uses the continuous-time limit: for absorbing diffusion,
        the ELBO at time t equals CE(x, x̂(x_t, z₀, t)) / (1 − t + ε),
        so the importance-weighted loss with reweighting w(t) becomes:
            L = E_t [ w(λ(t)) · CE / (1−t) ]

        We approximate λ(t) = log(1/(t+ε)) as a simple mask-rate SNR.
        """
        B = tokens.size(0)
        # Sample t avoiding t=1 (fully masked → degenerate)
        t = torch.rand(B, device=tokens.device) * 0.95 + 0.025  # (B,) ∈ [0.025, 0.975]

        x_t, mask = self._mask_tokens(tokens, t)

        logits = self.forward(x_t, z0, t)   # (B, L, V)

        # Cross-entropy only on masked positions
        ce = F.cross_entropy(
            logits[mask],
            tokens[mask],
            reduction="mean",
        ) if mask.any() else logits.new_zeros(())

        # SNR proxy for masked diffusion: λ(t) ≈ log((1-t)/t)
        lam_t = torch.log((1 - t + 1e-6) / (t + 1e-6))         # (B,)
        w     = self.loss_factor * sigmoid_weight(lam_t, self.sigmoid_bias)  # (B,)

        return w.mean() * ce

    @torch.no_grad()
    def sample(self, z0: Tensor, n_steps: int = 20) -> Tensor:
        B, L = z0.size(0), self.seq_len
        x = torch.full((B, L), MASK_ID, dtype=torch.long, device=z0.device)

        for i in range(n_steps, 0, -1):
            t = torch.full((B,), i / n_steps, device=z0.device)
            logits = self.forward(x, z0, t)
            probs = torch.softmax(logits, dim=-1)

            sampled = torch.multinomial(probs.reshape(-1, self.vocab_size), 1).reshape(B, L)

            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            n_unmask = max(1, round(L * (1 - (i - 1) / n_steps)))

            still_masked = (x == MASK_ID)
            confidence[~still_masked] = -1.0

            topk_idx = confidence.topk(min(n_unmask, still_masked.sum(dim=-1).max().item()), dim=-1).indices
            
            commit_mask = torch.zeros_like(still_masked)
            commit_mask.scatter_(-1, topk_idx, True)
            commit_mask &= still_masked
            
            x[commit_mask] = sampled[commit_mask]

        if (x == MASK_ID).any():
            t = torch.full((B,), 1 / n_steps, device=z0.device)
            logits = self.forward(x, z0, t)
            x[x == MASK_ID] = logits.argmax(-1)[x == MASK_ID]

        return x