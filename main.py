from __future__ import annotations

import argparse
import random
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import alpha_sigma_from_logsnr
from trainer import ULConfig, ULTrainer, load_trainer, TinyStoriesDataset, GSM8KDataset


def _reconstruction_check(trainer: ULTrainer, dataset: Dataset, device: str, n: int = 4, n_steps: int = 30, decode_text: bool = False):
    trainer.vae.eval()
    trainer.decoder.eval()

    test_tokens = torch.stack([dataset[i] for i in range(n)]).to(device)

    with torch.no_grad():
        z_clean = trainer.vae.encode(test_tokens)
        lam0    = torch.full((n,), trainer.cfg.lam_min_fixed, device=device)
        a0, s0  = alpha_sigma_from_logsnr(lam0)
        z0      = a0[:, None] * z_clean + s0[:, None] * torch.randn_like(z_clean)
        recon   = trainer.decoder.sample(z0, seq_len=trainer.cfg.seq_len, n_steps=n_steps)

    print("Original:")
    for row in test_tokens.tolist():
        if decode_text:
            print(f"  {GSM8KDataset.decode(row, model_name=trainer.cfg.model_name)!r}")
        else:
            print(f"  {row}")

    print("Reconstructed:")
    for row in recon.tolist():
        if decode_text:
            print(f"  {GSM8KDataset.decode(row, model_name=trainer.cfg.model_name)!r}")
        else:
            print(f"  {row}")

    acc = (recon == test_tokens).float().mean().item()
    print(f"\nToken accuracy on {n} examples: {acc:.2%}")


def _generation_samples(trainer: ULTrainer, device: str, n: int = 8, n_prior_steps: int = 50, n_dec_steps: int = 20, decode_text: bool = False):
    samples = trainer.generate(n=n, n_prior_steps=n_prior_steps, n_dec_steps=n_dec_steps)
    for i, seq in enumerate(samples):
        if decode_text:
            print(f"  Sample {i+1}: {GSM8KDataset.decode(seq, model_name=trainer.cfg.model_name)!r}")
        else:
            print(f"  Sample {i+1}: {seq}")


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Train Unified Latents on TinyStories, or GSM8K")

    # --- dataset ---
    parser.add_argument("--dataset", choices=["tinystories", "gsm8k"], default="gsm8k", help="Dataset to train on (default: gsm8k)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of GSM8K examples loaded "
                             "(default: all ~7.5k train / 1.3k test)")
    parser.add_argument("--gsm8k-split", default="train", choices=["train", "test"], help="GSM8K split to use (default: train)")
    parser.add_argument("--max-stories", type=int, default=None, help="Cap number of TinyStories stories loaded (default: all)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--n-samples", type=int, default=8192, help="Number of samples for toy dataset (default: 8192)")

    # --- sequence ---
    parser.add_argument("--seq-len", type=int, default=128, help="Token sequence length (default: 128)")

    # --- Qwen3 backbone ---
    parser.add_argument("--model-name", default="Qwen/Qwen3-2B", help="HuggingFace model id for encoder and decoder (default: Qwen/Qwen3-2B)")
    parser.add_argument("--n-encoder-layers", type=int, default=8, help="Number of Qwen3 layers to use as encoder (default: 8)")
    parser.add_argument("--n-latent-tokens", type=int, default=8, help="Number of z0 prefix tokens for decoder conditioning (default: 8)")
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Freeze Qwen3 encoder backbone (default: True)")
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false", help="Unfreeze encoder for full fine-tuning")
    parser.add_argument("--freeze-decoder", action="store_true", default=False, help="Freeze Qwen3 decoder backbone (default: False)")

    # --- latent / prior ---
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent vector dimension (default: 256)")
    parser.add_argument("--prior-d-model", type=int, default=512, help="Prior d_model (default: 512)")
    parser.add_argument("--prior-n-layers", type=int, default=4, help="Prior transformer layers (default: 4)")

    # --- CE warm-up decoder ---
    parser.add_argument("--dec-d-model", type=int, default=512, help="CE warm-up decoder d_model (default: 512)")
    parser.add_argument("--dec-n-layers", type=int, default=2, help="CE warm-up decoder layers (default: 2)")

    # --- UL hyperparameters ---
    parser.add_argument("--lam-min-fixed", type=float, default=5.0)
    parser.add_argument("--sigmoid-bias", type=float, default=0.0)
    parser.add_argument("--loss-factor", type=float, default=1.5)

    # --- training ---
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR linear warm-up steps (default: 1000)")
    parser.add_argument("--log-every", type=int, default=100)

    # --- checkpoint / eval ---
    parser.add_argument("--checkpoint", default=None, help="Load a saved checkpoint instead of training from scratch")
    parser.add_argument("--save-path", default="model_trainer.pth", help="Where to save the checkpoint (default: model_trainer.pth)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only run generation and reconstruction checks")

    # --- sampling ---
    parser.add_argument("--n-prior-steps", type=int, default=50)
    parser.add_argument("--n-dec-steps", type=int, default=20)
    parser.add_argument("--n-gen-samples", type=int, default=8)

    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    decode_text = False

    if args.dataset == "tinystories":
        seq_len = args.seq_len
        print(f"TinyStories  |  seq_len={seq_len}")
        dataset = TinyStoriesDataset(
            seq_len = seq_len,
            split = "train",
            max_stories = args.max_stories,
            cache_dir = args.cache_dir,
        )
        vocab_size  = dataset.vocab_size
        decode_text = True
        print(f"  vocab_size={vocab_size:,}  chunks={len(dataset):,}\n")

    else: # gsm8k
        seq_len = args.seq_len
        print(f"GSM8K  |  split={args.gsm8k_split}  seq_len={seq_len}")
        dataset = GSM8KDataset(
            seq_len = seq_len,
            split = args.gsm8k_split,
            max_samples = args.max_samples,
            cache_dir = args.cache_dir,
            model_name = args.model_name,
        )
        vocab_size = dataset.vocab_size
        decode_text = True
        print(f"  vocab_size={vocab_size:,}  chunks={len(dataset):,}\n")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    cfg = ULConfig(
        vocab_size = vocab_size,
        seq_len = seq_len,
        model_name = args.model_name,
        n_encoder_layers = args.n_encoder_layers,
        freeze_encoder = args.freeze_encoder,
        freeze_decoder = args.freeze_decoder,
        n_latent_tokens = args.n_latent_tokens,
        latent_dim = args.latent_dim,
        prior_d_model = args.prior_d_model,
        prior_n_layers = args.prior_n_layers,
        lam_min_fixed = args.lam_min_fixed,
        dec_d_model = args.dec_d_model,
        dec_n_layers = args.dec_n_layers,
        sigmoid_bias = args.sigmoid_bias,
        loss_factor = args.loss_factor,
        lr = args.lr,
        batch_size = args.batch_size,
        n_epochs = args.n_epochs,
        stage2_epochs = args.stage2_epochs,
        warmup_steps = args.warmup_steps,
        log_every = args.log_every,
    )

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint} …")
        trainer = load_trainer(args.checkpoint, device=device)
        cfg     = trainer.cfg
    else:
        trainer = ULTrainer(cfg, device=device)

    print(f"VAE params: {_count_params(trainer.vae):,}")
    print(f"Prior params {_count_params(trainer.prior):,}")
    print(f"Decoder params {_count_params(trainer.decoder):,}")
    print(f"Base model params: {_count_params(trainer.base_model):,}")
    print()

    if not args.eval_only:
        trainer.train(loader)

        print(f"\nSaving checkpoint to {args.save_path} …")
        torch.save({
            "cfg": trainer.cfg,
            "vae": trainer.vae.state_dict(),
            "prior": trainer.prior.state_dict(),
            "decoder": trainer.decoder.state_dict(),
            "base_model": trainer.base_model.state_dict(),
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