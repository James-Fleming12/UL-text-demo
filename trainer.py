from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

from datasets import load_dataset
from model import TextVAE, MaskedTextDecoder, LatentDiffusionPrior, alpha_sigma_from_logsnr, MASK_ID

@dataclass
class ULConfig:
    # tokeniser / data
    vocab_size: int = 152001   # Qwen3 vocab size + 1 for MASK
    seq_len: int = 128

    # Qwen3 backbone
    model_name: str = "Qwen/Qwen3-2B"
    n_encoder_layers: int = 8
    freeze_encoder: bool = True
    freeze_decoder: bool = False
    n_latent_tokens: int = 8

    # latent / prior
    latent_dim: int = 256
    prior_d_model: int = 512
    prior_n_layers: int = 4
    lam_min_fixed: float = 5.0

    # CE warm-up decoder (lightweight)
    dec_d_model: int   = 512
    dec_n_layers: int   = 2

    # UL hyperparameters
    sigmoid_bias: float = 0.0
    loss_factor: float = 1.5

    # training
    lr: float = 3e-4
    batch_size: int = 128
    n_epochs: int = 10
    stage2_epochs: int = 5
    log_every: int = 100
    warmup_steps: int = 1000

class TinyStoriesDataset(Dataset):
    """
    HuggingFace roneneldan/TinyStories tokenised with GPT-2 BPE.
    Token 0 is reserved for [MASK]; all GPT-2 ids are shifted up by 1.
    vocab_size = tokeniser.vocab_size + 1.
    """

    def __init__(
        self,
        seq_len     : int           = 128,
        split       : str           = "train",
        max_stories : Optional[int] = None,
        cache_dir   : Optional[str] = None,
    ):
        try:
            from datasets import load_dataset
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ImportError("pip install datasets transformers")

        self.seq_len = seq_len
        raw = load_dataset("roneneldan/TinyStories", split=split, cache_dir=cache_dir)
        if max_stories is not None:
            raw = raw.select(range(min(max_stories, len(raw))))

        print(f"  Tokenising {len(raw):,} stories …")
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.model_max_length = int(1e9)
        tok.pad_token        = tok.eos_token
        self.vocab_size      = tok.vocab_size + 1
        self.pad_id          = tok.eos_token_id + 1

        self.chunks: List[List[int]] = []
        for ex in raw:
            ids = [i + 1 for i in tok.encode(ex["text"], add_special_tokens=False)]
            for start in range(0, max(1, len(ids)), seq_len):
                chunk = ids[start : start + seq_len]
                if len(chunk) < seq_len:
                    chunk += [self.pad_id] * (seq_len - len(chunk))
                self.chunks.append(chunk)

        print(f"  {len(self.chunks):,} chunks of length {seq_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)

    @staticmethod
    def decode(ids: List[int], skip_special: bool = True) -> str:
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            return str(ids)
        tok      = GPT2TokenizerFast.from_pretrained("gpt2")
        real_ids = [i - 1 for i in ids if 1 <= i <= tok.vocab_size]
        return tok.decode(real_ids, skip_special_tokens=skip_special)

class GSM8KDataset(Dataset):
    """
    HuggingFace openai/gsm8k tokenised with the Qwen3 tokeniser.

    Each example is formatted as:
        "Question: {question}\\nAnswer: {answer}"
    then tokenised and chunked / padded to seq_len tokens.

    Token 0 is reserved for [MASK]; all Qwen3 ids are shifted up by 1.
    vocab_size = tokeniser.vocab_size + 1.

    Args:
        seq_len     : fixed sequence length for all chunks
        split       : "train" or "test"
        max_samples : cap the number of GSM8K examples loaded
                      (None = use all ~7.5k train / 1.3k test)
        cache_dir   : HuggingFace cache directory
        model_name  : tokeniser source — should match the decoder backbone
    """

    def __init__(
        self,
        seq_len: int = 128,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-2B",
    ):
        self.seq_len = seq_len

        print(f"  Loading GSM8K ({split}) …")
        raw = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))
        print(f"  {len(raw):,} examples loaded")

        print(f"  Tokenising with {model_name} tokeniser …")
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.model_max_length = int(1e9)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        self.vocab_size = tok.vocab_size + 1   # +1 reserves 0 for [MASK]
        self.pad_id = tok.eos_token_id + 1

        self.chunks: List[List[int]] = []
        for ex in raw:
            text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
            ids = [i + 1 for i in tok.encode(text, add_special_tokens=False)]
            for start in range(0, max(1, len(ids)), seq_len):
                chunk = ids[start : start + seq_len]
                if len(chunk) < seq_len:
                    chunk += [self.pad_id] * (seq_len - len(chunk))
                self.chunks.append(chunk)

        print(f"  {len(self.chunks):,} chunks of length {seq_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor(self.chunks[idx], dtype=torch.long)

    @staticmethod
    def decode(ids: List[int], model_name: str = "Qwen/Qwen3-2B") -> str:
        tok = AutoTokenizer.from_pretrained(model_name)
        real_ids = [i - 1 for i in ids if 1 <= i <= tok.vocab_size]
        return tok.decode(real_ids, skip_special_tokens=True)

class ULTrainer:
    """
    Stage 1 : jointly train TextVAE encoder projection + prior + Qwen3 decoder.
              L = L_prior(z, w=1) + L_decoder(x, sigmoid) + 0.1·L_ce
    Stage 2 : freeze encoder; train a larger base prior with sigmoid weighting.
    """
    def __init__(self, cfg: ULConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)

        self.vae = TextVAE(
            model_name = cfg.model_name,
            latent_dim = cfg.latent_dim,
            n_encoder_layers = cfg.n_encoder_layers,
            freeze_backbone = cfg.freeze_encoder,
            vocab_size = cfg.vocab_size,
            seq_len = cfg.seq_len,
            dec_d_model = cfg.dec_d_model,
            dec_n_layers = cfg.dec_n_layers,
        ).to(self.device)

        self.prior = LatentDiffusionPrior(
            latent_dim = cfg.latent_dim,
            d_model = cfg.prior_d_model,
            n_layers = cfg.prior_n_layers,
            lam_min_fixed = cfg.lam_min_fixed,
        ).to(self.device)

        self.decoder = MaskedTextDecoder(
            model_name = cfg.model_name,
            latent_dim = cfg.latent_dim,
            n_latent_tokens = cfg.n_latent_tokens,
            sigmoid_bias = cfg.sigmoid_bias,
            loss_factor = cfg.loss_factor,
            freeze_backbone = cfg.freeze_decoder,
            vocab_size = cfg.vocab_size,
        ).to(self.device)

        self.base_model = LatentDiffusionPrior(
            latent_dim = cfg.latent_dim,
            d_model = cfg.prior_d_model * 2,
            n_layers = cfg.prior_n_layers * 2,
            lam_min_fixed = cfg.lam_min_fixed,
        ).to(self.device)

        s1_params = (
            [p for p in self.vae.parameters() if p.requires_grad]
            + [p for p in self.prior.parameters() if p.requires_grad]
            + [p for p in self.decoder.parameters() if p.requires_grad]
        )
        self.opt_stage1 = torch.optim.AdamW(s1_params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
        self.scheduler_s1 = torch.optim.lr_scheduler.LinearLR(
            self.opt_stage1, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_steps,
        )

        self.opt_stage2 = torch.optim.AdamW(
            self.base_model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1,
        )
        self.scheduler_s2 = torch.optim.lr_scheduler.LinearLR(
            self.opt_stage2, start_factor=0.01, end_factor=1.0, total_iters=max(1, cfg.warmup_steps // 2),
        )

    def _stage1_step(self, tokens: Tensor) -> dict:
        tokens = tokens.to(self.device)
        z_clean = self.vae.encode(tokens)

        lam0 = torch.full((tokens.size(0),), self.cfg.lam_min_fixed, device=self.device)
        a0, s0 = alpha_sigma_from_logsnr(lam0)
        z0 = a0[:, None] * z_clean + s0[:, None] * torch.randn_like(z_clean)

        l_prior = self.prior.loss(z_clean)
        l_dec = self.decoder.loss(tokens, z0)
        l_rec = self.vae.reconstruction_loss(tokens)
        loss = l_prior + l_dec + 0.1 * l_rec

        self.opt_stage1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(self.vae.parameters()) + list(self.prior.parameters()) + list(self.decoder.parameters()) if p.requires_grad],
            1.0,
        )
        self.opt_stage1.step()
        self.scheduler_s1.step()

        return {"loss": loss.item(), "l_prior": l_prior.item(), "l_dec": l_dec.item(), "l_rec": l_rec.item()}

    def _stage2_step(self, tokens: Tensor, epoch: int, warmup_epochs: int = 2) -> dict:
        """Warm up with unweighted loss for `warmup_epochs` to prevent collapse."""
        tokens = tokens.to(self.device)
        with torch.no_grad():
            z_clean = self.vae.encode(tokens)

        loss = self.base_model.loss(z_clean) if epoch <= warmup_epochs else self.base_model.sigmoid_loss(z_clean)

        self.opt_stage2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
        self.opt_stage2.step()
        self.scheduler_s2.step()

        return {"loss_s2": loss.item()}

    def train(self, loader: DataLoader):
        cfg  = self.cfg
        step = 0

        print("=" * 60)
        print("STAGE 1: Joint encoder + prior + decoder training")
        print("=" * 60)
        for epoch in range(1, cfg.n_epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                tokens = batch[0] if isinstance(batch, (list, tuple)) else batch
                m = self._stage1_step(tokens)
                epoch_loss += m["loss"]
                step += 1
                if step % cfg.log_every == 0:
                    print(f"  step {step:5d} | loss={m['loss']:.4f} l_prior={m['l_prior']:.4f} l_dec={m['l_dec']:.4f} l_rec={m['l_rec']:.4f}")
            print(f"Epoch {epoch}/{cfg.n_epochs} — avg loss: {epoch_loss / len(loader):.4f}")

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
                tokens = batch[0] if isinstance(batch, (list, tuple)) else batch
                m = self._stage2_step(tokens, epoch=epoch)
                epoch_loss += m["loss_s2"]
                step2      += 1
                if step2 % cfg.log_every == 0:
                    print(f"  step {step2:5d} | loss_s2={m['loss_s2']:.4f}")
            print(f"Epoch {epoch}/{cfg.stage2_epochs} — avg loss: {epoch_loss / len(loader):.4f}")

    @torch.no_grad()
    def generate(self, n: int = 4, n_prior_steps: int = 50, n_dec_steps: int = 20) -> List[List[int]]:
        self.base_model.eval()
        self.decoder.eval()
        z1 = torch.randn(n, self.cfg.latent_dim, device=self.device)
        z0 = self.base_model.sample_z0(z1, n_steps=n_prior_steps)
        tokens = self.decoder.sample(z0, seq_len=self.cfg.seq_len, n_steps=n_dec_steps)
        return tokens.cpu().tolist()

def load_trainer(path: str, device: str = "cpu") -> ULTrainer:
    ckpt = torch.load(path, map_location=device)
    trainer = ULTrainer(ckpt["cfg"], device=device)
    trainer.vae.load_state_dict(ckpt["vae"])
    trainer.prior.load_state_dict(ckpt["prior"])
    trainer.decoder.load_state_dict(ckpt["decoder"])
    trainer.base_model.load_state_dict(ckpt["base_model"])
    return trainer