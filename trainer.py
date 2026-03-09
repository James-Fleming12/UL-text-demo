from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import TextVAE, MaskedTextDecoder, LatentDiffusionPrior, alpha_sigma_from_logsnr, sigmoid_weight

@dataclass
class ULConfig:
    # tokeniser / data
    vocab_size   : int   = 32
    seq_len      : int   = 16
    # encoder / VAE
    vae_d_model  : int   = 64
    vae_n_layers : int   = 2
    latent_dim   : int   = 32
    # prior
    prior_d_model: int   = 64
    prior_n_layers: int  = 2
    lam_min_fixed: float = 5.0
    # decoder
    dec_d_model  : int   = 64
    dec_n_layers : int   = 2
    sigmoid_bias : float = 0.0
    loss_factor  : float = 1.5
    # training
    lr           : float = 3e-4
    batch_size   : int   = 64
    n_epochs     : int   = 20
    stage2_epochs: int   = 10
    log_every    : int   = 50


class ULTrainer:
    """
    Orchestrates stage-1 and stage-2 training of Unified Latents.

    Stage 1: jointly train encoder (TextVAE), prior, and masked decoder.
             Objective: L = L_prior(z) + L_decoder(x)
             (The VAE CE-decoder loss is optionally included for warm-up.)

    Stage 2: freeze encoder, retrain prior with sigmoid weighting (acts as the
             generative base model).  The decoder is also frozen in stage 2.
    """

    def __init__(self, cfg: ULConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)

        self.vae = TextVAE(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            d_model=cfg.vae_d_model,
            n_layers=cfg.vae_n_layers,
            latent_dim=cfg.latent_dim,
        ).to(self.device)

        self.prior = LatentDiffusionPrior(
            latent_dim=cfg.latent_dim,
            d_model=cfg.prior_d_model,
            n_layers=cfg.prior_n_layers,
            lam_min_fixed=cfg.lam_min_fixed,
        ).to(self.device)

        self.decoder = MaskedTextDecoder(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.seq_len,
            d_model=cfg.dec_d_model,
            n_layers=cfg.dec_n_layers,
            latent_dim=cfg.latent_dim,
            sigmoid_bias=cfg.sigmoid_bias,
            loss_factor=cfg.loss_factor,
        ).to(self.device)

        # Stage-1 optimiser: all three components
        self.opt_stage1 = torch.optim.AdamW(
            list(self.vae.parameters())
            + list(self.prior.parameters())
            + list(self.decoder.parameters()),
            lr=cfg.lr,
        )

        # Stage-2 base model: a *new* prior trained with sigmoid weighting on
        # frozen encoder latents.  Here we reuse the same architecture but
        # reinitialise weights and train with a different loss weighting.
        self.base_model = LatentDiffusionPrior(
            latent_dim=cfg.latent_dim,
            d_model=cfg.prior_d_model * 2,   # slightly larger, as in paper
            n_layers=cfg.prior_n_layers * 2,
            lam_min_fixed=cfg.lam_min_fixed,
        ).to(self.device)

        self.opt_stage2 = torch.optim.AdamW(
            self.base_model.parameters(), lr=cfg.lr
        )

    def _stage1_step(self, tokens: Tensor) -> dict:
        tokens = tokens.to(self.device)

        z_clean = self.vae.encode(tokens)

        lam0    = torch.full((tokens.size(0),), self.cfg.lam_min_fixed, device=self.device)
        a0, s0  = alpha_sigma_from_logsnr(lam0)
        eps_z   = torch.randn_like(z_clean)
        z0      = a0[:, None] * z_clean + s0[:, None] * eps_z

        l_prior = self.prior.loss(z_clean)

        l_dec   = self.decoder.loss(tokens, z0)

        l_rec   = self.vae.reconstruction_loss(tokens)

        loss = l_prior + l_dec + 0.1 * l_rec

        self.opt_stage1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.vae.parameters())
            + list(self.prior.parameters())
            + list(self.decoder.parameters()),
            1.0,
        )
        self.opt_stage1.step()

        return {
            "loss": loss.item(),
            "l_prior": l_prior.item(),
            "l_dec": l_dec.item(),
            "l_rec": l_rec.item(),
        }

    def _stage2_step(self, tokens: Tensor) -> dict:
        """Stage 2: train base_model with sigmoid reweighting on frozen encoder."""
        tokens = tokens.to(self.device)

        with torch.no_grad():
            z_clean = self.vae.encode(tokens)

        # Sigmoid-weighted diffusion loss (better generative prior)
        B  = z_clean.size(0)
        t  = torch.rand(B, device=self.device)
        lam = self.base_model._logsnr_schedule(t)
        a, s = alpha_sigma_from_logsnr(lam)

        eps   = torch.randn_like(z_clean)
        z_t   = a[:, None] * z_clean + s[:, None] * eps
        z_hat = self.base_model(z_t, t)

        w    = sigmoid_weight(lam, b=0.0)
        dlam = torch.full_like(lam, self.base_model.lam_max_train
                               - self.base_model.lam_min_fixed)
        weight = w * dlam * torch.exp(lam / 2)
        loss   = (weight * ((z_clean - z_hat) ** 2).mean(dim=-1)).mean()

        self.opt_stage2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
        self.opt_stage2.step()

        return {"loss_s2": loss.item()}

    def train(self, loader: DataLoader):
        cfg = self.cfg
        step = 0

        print("=" * 60)
        print("STAGE 1: Joint encoder + prior + decoder training")
        print("=" * 60)
        for epoch in range(1, cfg.n_epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    tokens = batch[0]
                else:
                    tokens = batch
                metrics = self._stage1_step(tokens)
                epoch_loss += metrics["loss"]
                step += 1
                if step % cfg.log_every == 0:
                    print(
                        f"  step {step:5d} | "
                        f"loss={metrics['loss']:.4f} "
                        f"l_prior={metrics['l_prior']:.4f} "
                        f"l_dec={metrics['l_dec']:.4f} "
                        f"l_rec={metrics['l_rec']:.4f}"
                    )
            print(f"Epoch {epoch}/{cfg.n_epochs} — avg loss: "
                  f"{epoch_loss / len(loader):.4f}")

        print()
        print("=" * 60)
        print("STAGE 2: Base model (larger prior) on frozen encoder")
        print("=" * 60)
        self.vae.eval()
        self.decoder.eval()
        step2 = 0
        for epoch in range(1, cfg.stage2_epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    tokens = batch[0]
                else:
                    tokens = batch
                metrics = self._stage2_step(tokens)
                epoch_loss += metrics["loss_s2"]
                step2 += 1
                if step2 % cfg.log_every == 0:
                    print(f"  step {step2:5d} | loss_s2={metrics['loss_s2']:.4f}")
            print(f"Epoch {epoch}/{cfg.stage2_epochs} — avg loss: "
                  f"{epoch_loss / len(loader):.4f}")

    @torch.no_grad()
    def generate(self, n: int = 4, n_prior_steps: int = 50,
                 n_dec_steps: int = 20) -> List[List[int]]:
        """
        Full generation pipeline:
          1. Sample z₁ ~ N(0, I)
          2. Denoise with base_model: z₁ → z₀
          3. Decode with masked decoder: z₀ → x
        """
        self.base_model.eval()
        self.decoder.eval()

        z1 = torch.randn(n, self.cfg.latent_dim, device=self.device)
        z0 = self.base_model.sample_z0(z1, n_steps=n_prior_steps)
        tokens = self.decoder.sample(z0, n_steps=n_dec_steps)
        return tokens.cpu().tolist()

class ToyTextDataset(Dataset):
    """
    Tiny synthetic dataset of token sequences drawn from a small vocabulary.
    Sequences follow simple patterns (repetitions, counting) to give the
    model something learnable.
    """

    VOCAB_SIZE = 32    # token ids 1..31 (0 reserved for MASK)

    def __init__(self, n_samples: int = 4096, seq_len: int = 16,
                 seed: int = 42):
        super().__init__()
        rng = random.Random(seed)
        self.data: List[List[int]] = []

        patterns = [
            self._repeat_pattern,
            self._count_pattern,
            self._alternate_pattern,
        ]
        for _ in range(n_samples):
            fn = rng.choice(patterns)
            self.data.append(fn(rng, seq_len))

    @staticmethod
    def _repeat_pattern(rng: random.Random, L: int) -> List[int]:
        tok = rng.randint(1, ToyTextDataset.VOCAB_SIZE - 1)
        return [tok] * L

    @staticmethod
    def _count_pattern(rng: random.Random, L: int) -> List[int]:
        start = rng.randint(1, ToyTextDataset.VOCAB_SIZE - L - 1)
        return [(start + i) % (ToyTextDataset.VOCAB_SIZE - 1) + 1 for i in range(L)]

    @staticmethod
    def _alternate_pattern(rng: random.Random, L: int) -> List[int]:
        a = rng.randint(1, ToyTextDataset.VOCAB_SIZE - 1)
        b = rng.randint(1, ToyTextDataset.VOCAB_SIZE - 1)
        return [a if i % 2 == 0 else b for i in range(L)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)

class TinyStoriesDataset(Dataset):
    """
    Streams roneneldan/TinyStories from HuggingFace and tokenises with
    the GPT-2 BPE tokeniser.

    Each item is a LongTensor of shape (seq_len,).  Stories shorter than
    seq_len are right-padded with the EOS token; longer stories are chunked
    into non-overlapping windows so no text is wasted.

    Token 0 ([MASK]) is *not* used by GPT-2, but the UL masked decoder
    reserves it.  We therefore shift all GPT-2 ids up by 1 so that
    id 0 remains free for [MASK].  vocab_size must be set to
    tokeniser.vocab_size + 1 when building the model.
    """

    def __init__(self, seq_len: int  = 64, split: str  = "train", max_stories: Optional[int] = None, cache_dir: Optional[str] = None):
        try:
            from datasets import load_dataset
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ImportError(
                "TinyStories mode requires the 'datasets' and 'transformers' "
                "packages.\n  pip install datasets transformers"
            )

        self.seq_len = seq_len

        print(f"  Loading TinyStories ({split}) …")
        raw = load_dataset("roneneldan/TinyStories", split=split, cache_dir=cache_dir)
        if max_stories is not None:
            raw = raw.select(range(min(max_stories, len(raw))))

        print(f"  Tokenising {len(raw):,} stories …")
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        self.vocab_size = tok.vocab_size + 1
        self.pad_id     = tok.eos_token_id + 1

        self.chunks: List[List[int]] = []
        for example in raw:
            ids = tok.encode(example["text"], add_special_tokens=False)
            ids = [i + 1 for i in ids]
            for start in range(0, max(1, len(ids)), seq_len):
                chunk = ids[start : start + seq_len]
                if len(chunk) < seq_len:
                    chunk += [self.pad_id] * (seq_len - len(chunk))
                self.chunks.append(chunk)

        print(f"  {len(self.chunks):,} chunks of length {seq_len}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)

    @staticmethod
    def decode(ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of (shifted) token ids back to a string."""
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            return str(ids)
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        real_ids = [i - 1 for i in ids if 1 <= i <= tok.vocab_size]
        return tok.decode(real_ids, skip_special_tokens=skip_special)

def load_trainer(path: str, device: str = "cpu") -> "ULTrainer":
    checkpoint = torch.load(path, map_location=device)
    trainer = ULTrainer(checkpoint["cfg"], device=device)
    trainer.vae.load_state_dict(checkpoint["vae"])
    trainer.prior.load_state_dict(checkpoint["prior"])
    trainer.decoder.load_state_dict(checkpoint["decoder"])
    trainer.base_model.load_state_dict(checkpoint["base_model"])
    return trainer