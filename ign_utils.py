"""Shared infrastructure: loss bookkeeping, DDP helpers, image saving.

Derived from IGN-main/util.py but with the distributed and wandb dependencies
made optional, so every script here runs unchanged on CPU, one GPU, or
torchrun.
"""

import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


# --------------------------------------------------------------------------
# distributed / device
# --------------------------------------------------------------------------

def ddp_setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"DDP rank {rank}/{dist.get_world_size()} up ({backend})")
    return rank


def ddp_cleanup(ddp):
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def resolve_device(args, rank=0):
    if getattr(args, "device", None):
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    # buffers (BatchNorm running stats) are copied outright
    ema_buffers = OrderedDict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        ema_buffers[name].copy_(buf)


# --------------------------------------------------------------------------
# frequency-matched noise (paper Section 4 / IGN-main util.py)
# --------------------------------------------------------------------------
# The paper never feeds white Gaussian noise on CelebA: it samples z whose
# per-frequency statistics match the data ("we apply an FFT to a batch of data
# and take the mean and variance for the real and imaginary parts of each
# frequency"). White noise has a flat spectrum and is spectrally far
# off-manifold; this colored noise is what the published CelebA results use.
# Ported from IGN-main/util.py::generate_noise_with_image_stats, split in two
# so the statistics can be saved in checkpoints and reused at experiment time.

def freq_stats_from_images(images):
    """Per-frequency mean/var of the batch's FFT, as a (C, H, W)-shaped dict."""
    f = torch.fft.fftn(images, dim=(-2, -1))
    return {"real_mean": f.real.mean(0), "real_var": f.real.var(0),
            "imag_mean": f.imag.mean(0), "imag_var": f.imag.var(0)}


def sample_freq_noise(stats, batch_size):
    """Sample real-valued noise whose spectrum follows `stats`.

    Mirrors the upstream construction: draw the top-left quarter of the
    frequency plane, mirror it (conjugate-symmetrically) to the full plane so
    the inverse FFT is (approximately) real, zero the DC/Nyquist imaginary
    parts, and take the real part.
    """
    C, H, W = stats["real_mean"].shape
    qH, qW = H // 2 + H % 2, W // 2 + W % 2

    def quarter(mean_key, var_key):
        mean = stats[mean_key][:, :qH, :qW].unsqueeze(0).expand(batch_size, C, qH, qW)
        std = stats[var_key][:, :qH, :qW].sqrt().unsqueeze(0).expand(batch_size, C, qH, qW)
        return torch.normal(mean, std)

    real_q = quarter("real_mean", "real_var")
    imag_q = quarter("imag_mean", "imag_var")
    imag_q[:, :, 0, 0] = 0  # DC component is real

    real_half = torch.cat((real_q, real_q[..., W % 2:].flip(-1)), dim=-1)
    imag_half = torch.cat((imag_q, -imag_q[..., W % 2:].flip(-1)), dim=-1)
    real_full = torch.cat((real_half, real_half[:, :, H % 2:, :].flip(-2)), dim=-2)
    imag_full = torch.cat((imag_half, -imag_half[:, :, H % 2:, :].flip(-2)), dim=-2)

    if H % 2 == 0:
        imag_full[:, :, H // 2, 0] = 0  # Nyquist frequency for height
    if W % 2 == 0:
        imag_full[:, :, 0, W // 2] = 0  # Nyquist frequency for width

    return torch.fft.ifftn(torch.complex(real_full, imag_full), dim=(-2, -1)).real


def generate_noise_with_image_stats(images):
    """One-shot version, same name and behaviour as upstream."""
    return sample_freq_noise(freq_stats_from_images(images), images.shape[0])


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

class LossManager:
    """Accumulates the three IGN losses, applies weights, steps the optimiser.

    Mirrors upstream's LossManager, including the tanh clamp on the
    discriminative term that keeps it from overwhelming loss_in.
    """

    def __init__(self, f, args, run_dir=None):
        self.weights = {
            "loss_in": args.in_manifold_loss_factor,
            "loss_out": args.out_of_manifold_loss_factor,
            "loss_out_d": args.discriminative_loss_factor,
        }
        self.opt = torch.optim.Adam(f.parameters(), lr=args.lr,
                                    betas=(args.beta1, args.beta2))
        self.scaler = torch.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())
        self.clamp_ratio = args.discriminative_loss_clamp_ratio
        self.clamp_val = args.discriminative_loss_clamp_val
        self.epoch_begin_idem = args.epoch_begin_idempotence_losses
        self.grad_clip = args.grad_clip
        self.f = f
        self.run_dir = run_dir
        self.cur = {}
        self._epoch_sums = {}
        self._epoch_count = 0
        self._last = {}

    def add(self, name, value):
        self.cur[name] = self.cur.get(name, 0) + value

    def clamp_d(self, raw_loss):
        """Bound the discriminative loss to a tanh-saturating envelope.

        x0 is the smaller of an absolute cap and a multiple of loss_in, so the
        term stays comparable to the reconstruction loss and cannot explode.
        """
        x0_ratio = float(self.cur["loss_in"].detach()) * self.clamp_ratio
        x0 = min(self.clamp_val, x0_ratio)
        x0 = max(x0, 1e-8)
        x = raw_loss.mean(dim=tuple(range(1, raw_loss.dim())))
        return (x / x0).tanh() * x0

    def total(self, loss_dict, epoch):
        if epoch < self.epoch_begin_idem:
            return loss_dict["loss_in"] * self.weights["loss_in"]
        return sum(v * self.weights[k] for k, v in loss_dict.items())

    def step(self, epoch):
        loss = self.total(self.cur, epoch)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.f.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        self._last = {k: float(v.detach()) for k, v in self.cur.items()}
        self._last["total"] = float(loss.detach())
        for k, v in self._last.items():
            self._epoch_sums[k] = self._epoch_sums.get(k, 0.0) + v
        self._epoch_count += 1
        self.cur = {}

    def epoch_means(self):
        if self._epoch_count == 0:
            return {}
        return {k: v / self._epoch_count for k, v in self._epoch_sums.items()}

    def last_str(self):
        return " ".join(f"{k}={v:.4f}" for k, v in self._last.items())

    def epoch_str(self):
        return " ".join(f"{k}={v:.4f}" for k, v in self.epoch_means().items())

    def reset_epoch(self):
        self._epoch_sums, self._epoch_count = {}, 0

    def save_ckpt(self, model, epoch, args, path, extra=None):
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "args": vars(args),
            "attr_spec": args.attr_spec,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def maybe_resume(self, f, resume_path, device):
        if not resume_path:
            return 0
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]
        target = f.module if hasattr(f, "module") else f
        target.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"resumed from {resume_path} @ epoch {ckpt['epoch']}")
        return ckpt["epoch"] + 1


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def to_uint8(x):
    """[-1, 1] float tensor -> [0, 255] uint8, matching the training transform."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def save_image_grid(tensor, path, nrow=8):
    from torchvision.utils import save_image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor.detach().cpu().clamp(-1, 1), path, nrow=nrow, normalize=True,
               value_range=(-1, 1))


def denorm_for_plot(x):
    """(C,H,W) in [-1,1] -> (H,W,C) numpy in [0,1] for matplotlib."""
    x = x.detach().cpu().clamp(-1, 1)
    return ((x + 1) / 2).permute(1, 2, 0).numpy()
