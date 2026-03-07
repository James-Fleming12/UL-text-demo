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

from model import alpha_sigma_from_logsnr
from trainer import ToyTextDataset, ULConfig, ULTrainer

def main():
    """
    Unified Latents (UL) for Text — based on Heek et al. 2026
    =========================================================
    Components
    ----------
    1. TextVAE          - standard encoder/decoder VAE that maps token sequences to
                        a continuous latent z_clean and back via an MSE/cross-entropy
                        reconstruction loss (no diffusion decoder yet).
    2. LatentDiffusionPrior - small transformer that learns p(z₀ | z₁) by denoising
                            Gaussian-corrupted latents (continuous diffusion in
                            latent space).  The prior's minimum noise level λ(0)
                            is fixed so that the KL reduces to a weighted MSE.
    3. MaskedTextDecoder    - absorbing / masked diffusion model in *token* space that
                            learns p(x | z₀).  At each training step a random
                            fraction of tokens is replaced with [MASK], and the model
                            predicts the original tokens conditioned on z₀.
    4. ULTrainer            - orchestrates stage-1 training (encoder + prior + decoder
                            jointly) and stage-2 training (frozen encoder, larger
                            base model).
    5. toy_dataset / main   - tiny synthetic dataset and entry point.

    Design notes
    ------------
    * We follow the three-component objective of UL:
        L = L_prior(z)  +  L_decoder(x)
    where L_prior is the unweighted diffusion ELBO on z and
    L_decoder is the *reweighted* (sigmoid) masked-diffusion ELBO on x.

    * For the prior we use a cosine log-SNR schedule with a fixed
    λ_min = λ(0) = 5, matching the paper's choice of σ₀ ≈ 0.08.

    * For the masked decoder we use the absorbing-diffusion schedule of
    Austin et al. (2021), where at time t a fraction t of tokens is masked.
    The (reweighted) loss is the standard cross-entropy on masked positions,
    weighted by sigmoid(λ_x(t) - b) as described in the paper.

    * All hyperparameters are chosen to be small so the whole script runs on CPU
    for demonstration; scale up d_model, n_layers, etc. for real use.
    """
    torch.manual_seed(0)
    random.seed(0)

    SEQ_LEN    = 16
    VOCAB_SIZE = ToyTextDataset.VOCAB_SIZE   # 32

    cfg = ULConfig(
        vocab_size    = VOCAB_SIZE,
        seq_len       = SEQ_LEN,
        vae_d_model   = 64,
        vae_n_layers  = 2,
        latent_dim    = 32,
        prior_d_model = 64,
        prior_n_layers= 2,
        lam_min_fixed = 5.0,
        dec_d_model   = 64,
        dec_n_layers  = 2,
        sigmoid_bias  = 0.0,
        loss_factor   = 1.5,
        lr            = 3e-4,
        batch_size    = 128,
        n_epochs      = 30,
        stage2_epochs = 10,
        log_every     = 100,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Vocabulary size: {VOCAB_SIZE}, Sequence length: {SEQ_LEN}\n")

    dataset = ToyTextDataset(n_samples=8192, seq_len=SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    trainer = ULTrainer(cfg, device=device)

    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"VAE params      : {count_params(trainer.vae):,}")
    print(f"Prior params    : {count_params(trainer.prior):,}")
    print(f"Decoder params  : {count_params(trainer.decoder):,}")
    print(f"Base model params: {count_params(trainer.base_model):,}")
    print()

    trainer.train(loader)

    print()
    print("=" * 60)
    print("GENERATION SAMPLES (token ids)")
    print("=" * 60)
    samples = trainer.generate(n=8, n_prior_steps=50, n_dec_steps=20)
    for i, seq in enumerate(samples):
        print(f"  Sample {i+1}: {seq}")

    print()
    print("=" * 60)
    print("RECONSTRUCTION CHECK")
    print("=" * 60)
    trainer.vae.eval()
    trainer.decoder.eval()

    test_tokens = torch.stack([dataset[i] for i in range(4)]).to(device)
    print("Original:")
    for row in test_tokens.tolist():
        print(f"  {row}")

    with torch.no_grad():
        z_clean = trainer.vae.encode(test_tokens)
        lam0    = torch.full((4,), cfg.lam_min_fixed, device=device)
        a0, s0  = alpha_sigma_from_logsnr(lam0)
        z0      = a0[:, None] * z_clean + s0[:, None] * torch.randn_like(z_clean)
        recon   = trainer.decoder.sample(z0, n_steps=30)

    print("Reconstructed:")
    for row in recon.tolist():
        print(f"  {row}")

    acc = (recon == test_tokens).float().mean().item()
    print(f"\nToken accuracy on 4 examples: {acc:.2%}")

    print("\nDone.")

    torch.save({
        "cfg"        : trainer.cfg,
        "vae"        : trainer.vae.state_dict(),
        "prior"      : trainer.prior.state_dict(),
        "decoder"    : trainer.decoder.state_dict(),
        "base_model" : trainer.base_model.state_dict(),
    }, "model_trainer.pth")

if __name__ == "__main__":
    main()