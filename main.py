from __future__ import annotations

import argparse
import random
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import alpha_sigma_from_logsnr
from trainer import ToyTextDataset, ULConfig, ULTrainer, load_trainer, TinyStoriesDataset

def _reconstruction_check(trainer: ULTrainer, dataset: Dataset, device: str, n: int = 4, n_steps: int = 30, decode_text: bool = False):
    trainer.vae.eval()
    trainer.decoder.eval()

    test_tokens = torch.stack([dataset[i] for i in range(n)]).to(device)

    with torch.no_grad():
        z_clean = trainer.vae.encode(test_tokens)
        lam0    = torch.full((n,), trainer.cfg.lam_min_fixed, device=device)
        a0, s0  = alpha_sigma_from_logsnr(lam0)
        z0      = a0[:, None] * z_clean + s0[:, None] * torch.randn_like(z_clean)
        recon   = trainer.decoder.sample(z0, n_steps=n_steps)

    print("Original:")
    for row in test_tokens.tolist():
        if decode_text:
            print(f"  {TinyStoriesDataset.decode(row)!r}")
        else:
            print(f"  {row}")

    print("Reconstructed:")
    for row in recon.tolist():
        if decode_text:
            print(f"  {TinyStoriesDataset.decode(row)!r}")
        else:
            print(f"  {row}")

    acc = (recon == test_tokens).float().mean().item()
    print(f"\nToken accuracy on {n} examples: {acc:.2%}")

def _generation_samples(trainer: ULTrainer, device: str, n: int = 8, n_prior_steps: int = 50, n_dec_steps: int = 20, decode_text: bool = False):
    samples = trainer.generate(n=n, n_prior_steps=n_prior_steps, n_dec_steps=n_dec_steps)
    for i, seq in enumerate(samples):
        if decode_text:
            print(f"  Sample {i+1}: {TinyStoriesDataset.decode(seq)!r}")
        else:
            print(f"  Sample {i+1}: {seq}")

def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="Train Unified Latents on toy data or TinyStories")

    parser.add_argument("--dataset", choices=["toy", "tinystories"], default="tinystories", help="Dataset to train on (default: tinystories)")
    parser.add_argument("--max-stories", type=int, default=None,
                        help="Cap number of TinyStories stories loaded "
                             "(useful for quick tests; default: all)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory for TinyStories")
    parser.add_argument("--n-samples", type=int, default=8192, help="Number of samples for toy dataset (default: 8192)")

    parser.add_argument("--seq-len", type=int, default=128,
                        help="Token sequence length (default: 128 for TinyStories, "
                             "16 recommended for toy)")

    parser.add_argument("--latent-dim", type=int, default=256, help="Latent vector dimension (default: 256)")
    parser.add_argument("--d-model", type=int, default=512, help="Transformer d_model for VAE and decoder (default: 512)")
    parser.add_argument("--n-layers", type=int, default=6, help="Transformer layers for VAE and decoder (default: 6)")
    parser.add_argument("--prior-d-model", type=int, default=512, help="Prior d_model (default: 512)")
    parser.add_argument("--prior-n-layers", type=int, default=4, help="Prior layers (default: 4)")

    parser.add_argument("--lam-min-fixed", type=float, default=5.0)
    parser.add_argument("--sigmoid-bias", type=float, default=0.0)
    parser.add_argument("--loss-factor", type=float, default=1.5)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--checkpoint", default=None, help="Load a saved checkpoint instead of training from scratch")
    parser.add_argument("--save-path", default="model_trainer.pth", help="Where to save the checkpoint (default: model_trainer.pth)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only run generation and reconstruction checks")

    parser.add_argument("--n-prior-steps", type=int, default=50)
    parser.add_argument("--n-dec-steps", type=int, default=20)
    parser.add_argument("--n-gen-samples", type=int, default=8)

    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    decode_text = False

    if args.dataset == "toy":
        seq_len = args.seq_len
        dataset = ToyTextDataset(n_samples=args.n_samples, seq_len=seq_len)
        vocab_size = ToyTextDataset.VOCAB_SIZE
        print(f"Toy dataset  |  vocab={vocab_size}  seq_len={seq_len}  "
              f"n={len(dataset):,}\n")

    else:
        seq_len = args.seq_len if args.seq_len != 16 else 64   # sensible default
        print(f"TinyStories  |  seq_len={seq_len}")
        dataset    = TinyStoriesDataset(
            seq_len = seq_len,
            split = "train",
            max_stories = args.max_stories,
            cache_dir = args.cache_dir,
        )
        vocab_size  = dataset.vocab_size
        decode_text = True
        print(f"  vocab_size={vocab_size:,}  chunks={len(dataset):,}\n")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    prior_d = args.prior_d_model  or args.d_model
    prior_n = args.prior_n_layers or args.n_layers

    cfg = ULConfig(
        vocab_size = vocab_size,
        seq_len = seq_len,
        vae_d_model = args.d_model,
        vae_n_layers = args.n_layers,
        latent_dim = args.latent_dim,
        prior_d_model = prior_d,
        prior_n_layers = prior_n,
        lam_min_fixed = args.lam_min_fixed,
        dec_d_model = args.d_model,
        dec_n_layers = args.n_layers,
        sigmoid_bias = args.sigmoid_bias,
        loss_factor = args.loss_factor,
        lr = args.lr,
        batch_size = args.batch_size,
        n_epochs = args.n_epochs,
        stage2_epochs = args.stage2_epochs,
        log_every = args.log_every,
    )

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint} …")
        trainer = load_trainer(args.checkpoint, device=device)
        cfg = trainer.cfg
    else:
        trainer = ULTrainer(cfg, device=device)

    print(f"VAE params : {_count_params(trainer.vae):,}")
    print(f"Prior params : {_count_params(trainer.prior):,}")
    print(f"Decoder params : {_count_params(trainer.decoder):,}")
    print(f"Base model params : {_count_params(trainer.base_model):,}")
    print()

    if not args.eval_only:
        trainer.train(loader)

        print(f"\nSaving checkpoint to {args.save_path} …")
        torch.save({
            "cfg" : trainer.cfg,
            "vae" : trainer.vae.state_dict(),
            "prior" : trainer.prior.state_dict(),
            "decoder" : trainer.decoder.state_dict(),
            "base_model" : trainer.base_model.state_dict(),
        }, args.save_path)
        print("Saved.\n")

    print("=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    _generation_samples(
        trainer,
        device,
        n = args.n_gen_samples,
        n_prior_steps = args.n_prior_steps,
        n_dec_steps = args.n_dec_steps,
        decode_text = decode_text,
    )

    print()
    print("=" * 60)
    print("RECONSTRUCTION CHECK")
    print("=" * 60)
    _reconstruction_check(
        trainer,
        dataset,
        device,
        n = 4,
        n_steps = args.n_dec_steps,
        decode_text = decode_text,
    )

    print("\nDone.")

if __name__ == "__main__":
    main()