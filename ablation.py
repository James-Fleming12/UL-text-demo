# called with python ul_ablation.py --checkpoint model_trainer.pth
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from model import TextVAE, MaskedTextDecoder, LatentDiffusionPrior, alpha_sigma_from_logsnr
from trainer import ULConfig, ULTrainer, ToyTextDataset, load_trainer

def _to_tokens(batch, device: torch.device) -> Tensor:
    t = batch[0] if isinstance(batch, (list, tuple)) else batch
    return t.to(device)


def _make_z0(z_clean: Tensor, lam_min_fixed: float) -> Tensor:
    """Add the fixed minimum noise z_clean → z₀  (matches UL convention)."""
    lam0   = torch.full((z_clean.size(0),), lam_min_fixed, device=z_clean.device)
    a0, s0 = alpha_sigma_from_logsnr(lam0)
    return a0[:, None] * z_clean + s0[:, None] * torch.randn_like(z_clean)

def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _train_vae(
    vae      : TextVAE,
    loader   : DataLoader,
    n_epochs : int,
    lr       : float,
    log_every: int,
    device   : torch.device,
) -> List[float]:
    """Train the TextVAE with plain teacher-forced cross-entropy.
    Returns per-step loss history."""
    vae = vae.to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    losses: List[float] = []
    step = 0

    print("  [Baseline A-1] VAE reconstruction training")
    for epoch in range(1, n_epochs + 1):
        vae.train()
        epoch_loss = 0.0
        for batch in loader:
            tokens = _to_tokens(batch, device)
            loss   = vae.reconstruction_loss(tokens)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
            step += 1
            if step % log_every == 0:
                print(f"    step {step:5d} | ce={loss.item():.4f}")
        print(f"    epoch {epoch}/{n_epochs}  avg={epoch_loss/len(loader):.4f}")

    return losses

def _train_decoder_on_frozen_vae(
    vae          : TextVAE,
    decoder      : MaskedTextDecoder,
    loader       : DataLoader,
    n_epochs     : int,
    lr           : float,
    lam_min_fixed: float,
    log_every    : int,
    device       : torch.device,
) -> List[float]:
    """Freeze the VAE encoder and train the MaskedTextDecoder on its latents.
    Returns per-step loss history."""
    vae.eval()
    decoder = decoder.to(device)
    opt     = torch.optim.AdamW(decoder.parameters(), lr=lr)
    losses: List[float] = []
    step = 0

    print("  [Baseline A-2] Masked decoder on frozen VAE latents")
    for epoch in range(1, n_epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        for batch in loader:
            tokens = _to_tokens(batch, device)
            with torch.no_grad():
                z_clean = vae.encode(tokens)
                z0      = _make_z0(z_clean, lam_min_fixed)
            loss = decoder.loss(tokens, z0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
            step += 1
            if step % log_every == 0:
                print(f"    step {step:5d} | dec={loss.item():.4f}")
        print(f"    epoch {epoch}/{n_epochs}  avg={epoch_loss/len(loader):.4f}")

    return losses

@torch.no_grad()
def _eval_vae_ce(
    vae   : TextVAE,
    loader: DataLoader,
    device: torch.device,
) -> float:
    vae.eval()
    total, n = 0.0, 0
    for batch in loader:
        tokens  = _to_tokens(batch, device)
        total  += vae.reconstruction_loss(tokens).item() * tokens.size(0)
        n      += tokens.size(0)
    return total / max(n, 1)

@torch.no_grad()
def _eval_decoder_loss(
    vae          : TextVAE,
    decoder      : MaskedTextDecoder,
    loader       : DataLoader,
    lam_min_fixed: float,
    device       : torch.device,
) -> float:
    vae.eval(); decoder.eval()
    total, n = 0.0, 0
    for batch in loader:
        tokens  = _to_tokens(batch, device)
        z_clean = vae.encode(tokens)
        z0      = _make_z0(z_clean, lam_min_fixed)
        total  += decoder.loss(tokens, z0).item() * tokens.size(0)
        n      += tokens.size(0)
    return total / max(n, 1)

@torch.no_grad()
def _eval_prior_loss(
    vae   : TextVAE,
    prior : LatentDiffusionPrior,
    loader: DataLoader,
    device: torch.device,
) -> float:
    vae.eval(); prior.eval()
    total, n = 0.0, 0
    for batch in loader:
        tokens  = _to_tokens(batch, device)
        z_clean = vae.encode(tokens)
        total  += prior.loss(z_clean).item() * tokens.size(0)
        n      += tokens.size(0)
    return total / max(n, 1)

@torch.no_grad()
def _eval_token_accuracy(
    vae          : TextVAE,
    decoder      : MaskedTextDecoder,
    loader       : DataLoader,
    lam_min_fixed: float,
    sample_steps : int,
    device       : torch.device,
) -> float:
    """Encode → z₀ → masked-decode, compare token-by-token to originals."""
    vae.eval(); decoder.eval()
    correct, total = 0, 0
    for batch in loader:
        tokens  = _to_tokens(batch, device)
        z_clean = vae.encode(tokens)
        z0      = _make_z0(z_clean, lam_min_fixed)
        recon   = decoder.sample(z0, n_steps=sample_steps)
        correct += (recon == tokens).sum().item()
        total   += tokens.numel()
    return correct / max(total, 1)

def _print_summary(results: Dict):
    ul = results["unified_latents"]
    bl = results["baseline"]

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    w_str = f"{bl['wall_time_s']:.1f}s" if bl["wall_time_s"] is not None else "N/A"

    print(f"\n{'='*64}")
    print(f"{'ABLATION RESULTS':^64}")
    print(f"{'='*64}")
    print(f"{'Metric':<26} {'Baseline (VAE→Dec)':>18} {'Unified Latents':>16}")
    print(f"{'-'*64}")
    print(f"{'token_acc':26} {fmt(bl['token_acc']):>18} {fmt(ul['token_acc']):>16}")
    print(f"{'avg_ce  (VAE CE-decoder)':26} {fmt(bl['avg_ce']):>18} {fmt(ul['avg_ce']):>16}")
    print(f"{'dec_loss (masked diffusion)':26} {fmt(bl['dec_loss']):>18} {fmt(ul['dec_loss']):>16}")
    print(f"{'prior_loss':26} {fmt(bl['prior_loss']):>18} {fmt(ul['prior_loss']):>16}")
    print(f"{'baseline train time':26} {w_str:>18} {'(pre-trained)':>16}")
    print(f"{'='*64}\n")

def run_ablation(
    checkpoint_path: str,
    dataset        : Dataset,
    vae_epochs     : int   = 20,
    dec_epochs     : int   = 20,
    batch_size     : int   = 128,
    lr             : float = 3e-4,
    val_fraction   : float = 0.1,
    sample_steps   : int   = 20,
    log_every      : int   = 100,
    device         : Optional[str] = None,
) -> Dict:
    """
    Load a pre-trained ULTrainer from `checkpoint_path`, train a fresh
    VAE→Decoder baseline on `dataset`, and compare both.

    The baseline VAE and decoder are built to match the architecture
    dimensions stored inside the checkpoint's ULConfig so that parameter
    counts are directly comparable.

    Parameters
    ----------
    checkpoint_path : path to the .pth file saved by main.py
    dataset         : any Dataset whose __getitem__ returns a LongTensor
                      of shape (seq_len,) with token 0 reserved for [MASK]
    vae_epochs      : epochs for baseline VAE training
    dec_epochs      : epochs for baseline decoder training
    batch_size      : batch size used for both training and evaluation
    lr              : learning rate for baseline optimiser
    val_fraction    : fraction of dataset held out for evaluation
    sample_steps    : unmasking steps used by token-accuracy evaluation
    log_every       : print a log line every N gradient steps
    device          : "cuda" / "cpu" / None (auto-detect)

    Returns
    -------
    dict with keys:
        "unified_latents" — metrics + "trainer" (the loaded ULTrainer)
        "baseline"        — metrics + "vae" and "decoder" models
        "config"          — the ULConfig from the checkpoint
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"\n{'='*64}")
    print(f"UL Ablation  |  checkpoint={checkpoint_path}  |  device={device}")
    print(f"{'='*64}\n")

    print("Loading pre-trained UL trainer …")
    ul_trainer = load_trainer(checkpoint_path, device=device)
    cfg: ULConfig = ul_trainer.cfg
    print(f"  Config: vocab={cfg.vocab_size}  seq_len={cfg.seq_len}  "
          f"latent_dim={cfg.latent_dim}")
    print(f"  UL params — "
          f"VAE={_count_params(ul_trainer.vae):,}  "
          f"Prior={_count_params(ul_trainer.prior):,}  "
          f"Decoder={_count_params(ul_trainer.decoder):,}  "
          f"BaseModel={_count_params(ul_trainer.base_model):,}\n")

    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    results: Dict = {"config": cfg}

    print("┌───────────────────────────────────────────────┐")
    print("│  Unified Latents  (loaded from checkpoint)     │")
    print("└───────────────────────────────────────────────┘")
    print("  Evaluating on held-out split …")

    ul_ce  = _eval_vae_ce(ul_trainer.vae, val_loader, dev)
    ul_dl  = _eval_decoder_loss(ul_trainer.vae, ul_trainer.decoder, val_loader, cfg.lam_min_fixed, dev)
    ul_pl  = _eval_prior_loss(ul_trainer.vae, ul_trainer.prior, val_loader, dev)
    ul_acc = _eval_token_accuracy(ul_trainer.vae, ul_trainer.decoder, val_loader, cfg.lam_min_fixed, sample_steps, dev)

    print(f"  token_acc={ul_acc:.4f}  avg_ce={ul_ce:.4f}  dec_loss={ul_dl:.4f}  prior_loss={ul_pl:.4f}\n")

    results["unified_latents"] = {
        "token_acc"  : ul_acc,
        "avg_ce"     : ul_ce,
        "dec_loss"   : ul_dl,
        "prior_loss" : ul_pl,
        "wall_time_s": None,
        "trainer"    : ul_trainer,
    }

    print("┌───────────────────────────────────────────────┐")
    print("│  Baseline: VAE  →  Frozen Masked Decoder      │")
    print("└───────────────────────────────────────────────┘")

    bl_vae = TextVAE(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.vae_d_model,
        n_layers=cfg.vae_n_layers,
        latent_dim=cfg.latent_dim,
    )
    bl_dec = MaskedTextDecoder(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        d_model=cfg.dec_d_model,
        n_layers=cfg.dec_n_layers,
        latent_dim=cfg.latent_dim,
        sigmoid_bias=cfg.sigmoid_bias,
        loss_factor=cfg.loss_factor,
    )
    print(f"  Baseline params — "
          f"VAE={_count_params(bl_vae):,}  "
          f"Decoder={_count_params(bl_dec):,}\n")

    t0 = time.time()
    _train_vae(bl_vae, train_loader, vae_epochs, lr, log_every, dev)
    _train_decoder_on_frozen_vae(
        bl_vae, bl_dec, train_loader, dec_epochs, lr,
        cfg.lam_min_fixed, log_every, dev,
    )
    wall_bl = time.time() - t0

    print("\n  Evaluating baseline on held-out split …")
    bl_ce  = _eval_vae_ce(bl_vae, val_loader, dev)
    bl_dl  = _eval_decoder_loss(bl_vae, bl_dec, val_loader, cfg.lam_min_fixed, dev)
    bl_acc = _eval_token_accuracy(bl_vae, bl_dec, val_loader,
                                   cfg.lam_min_fixed, sample_steps, dev)

    print(f"  token_acc={bl_acc:.4f}  avg_ce={bl_ce:.4f}  "
          f"dec_loss={bl_dl:.4f}  prior_loss=N/A\n")

    results["baseline"] = {
        "token_acc"  : bl_acc,
        "avg_ce"     : bl_ce,
        "dec_loss"   : bl_dl,
        "prior_loss" : None,
        "wall_time_s": wall_bl,
        "vae"        : bl_vae,
        "decoder"    : bl_dec,
    }

    _print_summary(results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a saved UL model against a fresh VAE→Decoder baseline"
    )
    parser.add_argument(
        "--checkpoint", default="model_trainer.pth",
        help="Path to .pth checkpoint saved by main.py  (default: model_trainer.pth)",
    )
    parser.add_argument("--n-samples",  type=int,   default=8192, help="Toy dataset size (default: 8192)")
    parser.add_argument("--vae-epochs", type=int,   default=20)
    parser.add_argument("--dec-epochs", type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--log-every",  type=int,   default=100)
    parser.add_argument("--device",     default=None, help="cuda / cpu  (default: auto-detect)")
    args = parser.parse_args()

    dataset = ToyTextDataset(n_samples=args.n_samples, seq_len=16)

    run_ablation(
        checkpoint_path = args.checkpoint,
        dataset         = dataset,
        vae_epochs      = args.vae_epochs,
        dec_epochs      = args.dec_epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        log_every       = args.log_every,
        device          = args.device,
    )