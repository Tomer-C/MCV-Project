"""Warm-start joint fine-tuning of a pair of IGNs to commute (f_A o f_B = f_B o f_A).

Motivation (see conversation / proposal): if f_A and f_B are idempotent AND
commute, then f_A o f_B is idempotent and Fix(f_A o f_B) = Fix(f_A) n Fix(f_B)
-- exactly the guarantee alternating projection needs but doesn't have for two
independently-trained, only-approximately-idempotent IGNs. This script does
NOT train f_A and f_B from scratch as one pipeline; it warm-starts from two
already-trained single-attribute checkpoints (cheap) and fine-tunes them
*jointly*, with a single optimizer stepping both nets from one combined loss,
adding a commutativity term on top of each net's own IGN losses:

    total = (loss_in_A + loss_out_A + loss_out_d_A)      -- keeps f_A anchored to Fix(f_A) = A
          + (loss_in_B + loss_out_B + loss_out_d_B)      -- keeps f_B anchored to Fix(f_B) = B
          + lambda_commute * D(f_A(f_B(x)), f_B(f_A(x))) -- pulls them toward commuting

Dropping the two per-net terms would let f_A = f_B trivially satisfy the
commutativity term, so they stay on by design -- see the conversation for why.
The commute term is evaluated on real images from an *unfiltered* pool
(--commute_spec, default "all") since Fix() is a statement about the whole
space, not either attribute's subset.

Usage
-----
  python -m train.train_commuting_pair \\
      --ckpt_a runs/Male/ckpt_last.pt --ckpt_b runs/Blond_Hair/ckpt_last.pt \\
      --run_name Male_Blond_commuting --num_epochs 30

Outputs land in <out_dir>/<run_name>/: ckpt_A_last.pt and ckpt_B_last.pt are
drop-in compatible with experiments/method1_single_ign/run_experiment.py --models (arch + attr_spec carried
in the checkpoint, same as train/train_ign.py's output).
"""

import argparse
import json
import os
import time

import torch

from ign.data import get_data_loader, spec_to_slug
from ign.models import ARCHS, load_ign
from ign.utils import (freq_stats_from_images, resolve_device, sample_freq_noise,
                       save_image_grid, set_seed)


def get_args_parser():
    p = argparse.ArgumentParser("Jointly fine-tune two IGNs to commute")

    p.add_argument("--ckpt_a", required=True, help="warm-start checkpoint for f_A")
    p.add_argument("--ckpt_b", required=True, help="warm-start checkpoint for f_B")
    p.add_argument("--attr_a", default=None,
                   help="attr_spec anchoring f_A's own losses; defaults to the "
                        "spec the checkpoint was trained with")
    p.add_argument("--attr_b", default=None, help="same as --attr_a, for f_B")
    p.add_argument("--commute_spec", default="all",
                   help="pool the commutativity term is evaluated on; 'all' = "
                        "unfiltered CelebA, since Fix(fA o fB)=Fix(fA)^Fix(fB) "
                        "is a whole-space statement, not an A- or B-only one")

    p.add_argument("--run_name", default=None)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--num_workers", default=2, type=int)

    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--num_epochs", default=30, type=int)
    p.add_argument("--steps_per_epoch", default=0, type=int,
                   help="0 = min length over the three loaders (A, B, commute)")
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--beta1", default=0.5, type=float)
    p.add_argument("--beta2", default=0.999, type=float)
    p.add_argument("--norm", default="auto", choices=["auto", "batch", "instance", "none"])
    p.add_argument("--grad_clip", action="store_true")
    p.add_argument("--seed", default=0, type=int)

    # per-network IGN loss weights -- same defaults/semantics as train/train_ign.py,
    # keep each net anchored to its own attribute so the commute term can't
    # just collapse f_A == f_B
    p.add_argument("--in_manifold_loss_factor", default=20, type=float)
    p.add_argument("--out_of_manifold_loss_factor", default=20, type=float)
    p.add_argument("--discriminative_loss_factor", default=2.5, type=float)
    p.add_argument("--discriminative_loss_clamp_ratio", default=1.5, type=float)
    p.add_argument("--discriminative_loss_clamp_val", default=9999.0, type=float)
    p.add_argument("--loss_p", default=1.0, type=float)
    p.add_argument("--noise_type", default="x_fft_stats",
                   choices=["x_fft_stats", "normal", "uniform"])

    # the new term
    p.add_argument("--lambda_commute", default=5.0, type=float,
                   help="weight of D(fA(fB(x)), fB(fA(x))); untuned, sweep it")
    p.add_argument("--commute_metric", default="l1", choices=["l1", "lpips"])

    p.add_argument("--device", default=None)
    p.add_argument("--save_every", default=5, type=int,
                   help="epochs; each save also keeps a ckpt_{A,B}_eNNNN.pt "
                        "snapshot (NNNN = epochs completed) so one long run "
                        "can be evaluated at several fine-tuning lengths")
    p.add_argument("--sample_every", default=1, type=int, help="epochs")
    p.add_argument("--log_every", default=50, type=int, help="steps")
    p.add_argument("--max_steps_per_epoch", default=0, type=int,
                   help=">0 truncates each epoch; used for smoke tests")
    return p


def make_noise(x, noise_type, freq_stats=None):
    if noise_type == "uniform":
        return torch.rand_like(x) * 2 - 1
    if noise_type == "x_fft_stats":
        return sample_freq_noise(freq_stats, x.shape[0])
    return torch.randn_like(x)


def infinite(loader):
    """Endless batches, reshuffled each pass. itertools.cycle would instead
    cache every batch it yields in RAM (~13 MB per 256-image batch)."""
    while True:
        yield from loader


def load_trainable_pair(ckpt_path, device, norm):
    """Load a trained IGN in train() mode plus a fresh f_copy snapshot for the
    idempotence loss's inner-f target (mirrors ign_models.build_ign, but
    warm-started from a checkpoint instead of random init)."""
    model, ckpt = load_ign(ckpt_path, device, norm=norm)
    model.train()
    arch_name = ckpt.get("args", {}).get("arch", "dcgan_enhanced")
    f_copy = ARCHS[arch_name](norm=norm).to(device)
    f_copy.load_state_dict(model.state_dict())
    f_copy.train()
    return model, f_copy, arch_name, ckpt


class NetLoss:
    """Per-network IGN losses (loss_in/loss_out/loss_out_d) + the tanh clamp
    on the discriminative term, factored out of LossManager so two nets can
    share one optimizer/backward instead of each owning its own."""

    def __init__(self, weights, clamp_ratio, clamp_val, loss_p):
        self.weights = weights
        self.clamp_ratio = clamp_ratio
        self.clamp_val = clamp_val
        self.loss_p = loss_p

    def clamp_d(self, raw_loss, loss_in_value):
        x0 = min(self.clamp_val, loss_in_value * self.clamp_ratio)
        x0 = max(x0, 1e-8)
        x = raw_loss.mean(dim=tuple(range(1, raw_loss.dim())))
        return (x / x0).tanh() * x0

    def __call__(self, f, f_copy, x, z):
        fx = f(x)
        fz = f(z)
        f_z = fz.detach()
        f_fz = f_copy(fz)
        ff_z = f(f_z)

        loss_in = (fx - x).abs().pow(self.loss_p).mean()
        loss_out = (f_fz - fz).abs().pow(self.loss_p).mean()
        raw_d = (ff_z - f_z).abs().pow(self.loss_p)
        loss_out_d = -self.clamp_d(raw_d, float(loss_in.detach())).mean()

        weighted = (self.weights["loss_in"] * loss_in
                    + self.weights["loss_out"] * loss_out
                    + self.weights["loss_out_d"] * loss_out_d)
        return weighted, {"loss_in": float(loss_in.detach()),
                          "loss_out": float(loss_out.detach()),
                          "loss_out_d": float(loss_out_d.detach())}


def commute_distance(metric_name, lpips_net):
    if metric_name == "l1":
        return lambda a, b: (a - b).abs().mean()

    def _lpips(a, b):
        return lpips_net(a, b).mean()
    return _lpips


def plot_curves(history, run_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def series(k):
        return [float("nan") if h.get(k) is None else h[k] for h in history]

    keys = [("loss_in_A", "loss_in A"), ("loss_in_B", "loss_in B"),
            ("loss_out_d_A", "loss_out_d A"), ("loss_out_d_B", "loss_out_d B"),
            ("commute", "commute D(fA(fB(x)), fB(fA(x)))"), ("total", "total")]
    xs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (k, title) in zip(axes.flat, keys):
        ys = series(k)
        ax.plot(xs, ys, marker=".", markersize=2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        vals = [v for v in ys if v == v]
        if vals and min(vals) > 0:
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_commute_grid(f_a, f_b, x, run_dir, epoch, n=8):
    """rows: x, fA(x), fB(x), fA(fB(x)), fB(fA(x)) -- the last two rows
    converging visually is the qualitative signal to watch alongside the
    commute loss curve. BN buffers snapshotted/restored (as train/train_ign.py does
    for its own sample grids): this is pure measurement, it must not perturb
    the running stats the training loop itself relies on."""
    buf_a = {k: v.clone() for k, v in f_a.named_buffers()}
    buf_b = {k: v.clone() for k, v in f_b.named_buffers()}
    x = x[:n]
    fb_x = f_b(x)
    fa_x = f_a(x)
    ab = f_a(fb_x)
    ba = f_b(fa_x)
    grid = torch.cat([x, fa_x, fb_x, ab, ba], 0)
    save_image_grid(grid, os.path.join(run_dir, "samples", f"e{epoch:04d}.png"), nrow=n)
    for k, v in f_a.named_buffers():
        v.copy_(buf_a[k])
    for k, v in f_b.named_buffers():
        v.copy_(buf_b[k])


def train(args):
    args.ddp = False  # single-GPU/CPU only; get_data_loader reads this
    device = resolve_device(args)
    set_seed(args.seed)

    ckpt_a0 = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    ckpt_b0 = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
    attr_a = args.attr_a or ckpt_a0.get("attr_spec")
    attr_b = args.attr_b or ckpt_b0.get("attr_spec")
    if not attr_a or not attr_b:
        raise ValueError("--attr_a/--attr_b not given and not found in the checkpoints; "
                         "pass them explicitly")

    if args.run_name is None:
        args.run_name = f"{spec_to_slug(attr_a)}_AND_{spec_to_slug(attr_b)}_commuting"
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as fh:
        json.dump(vars(args) | {"attr_a": attr_a, "attr_b": attr_b}, fh, indent=2)
    print(f"[{args.run_name}] device={device} A={attr_a!r} (warm from {args.ckpt_a}) "
          f"B={attr_b!r} (warm from {args.ckpt_b})")

    f_a, f_a_copy, arch_a, _ = load_trainable_pair(args.ckpt_a, device, args.norm)
    f_b, f_b_copy, arch_b, _ = load_trainable_pair(args.ckpt_b, device, args.norm)

    loader_a, _ = get_data_loader(args, attr_a, split="train")
    loader_b, _ = get_data_loader(args, attr_b, split="train")
    loader_c, _ = get_data_loader(args, args.commute_spec, split="train")
    print(f"[{args.run_name}] batches/epoch: A={len(loader_a)} B={len(loader_b)} "
          f"commute={len(loader_c)}")
    steps_per_epoch = args.steps_per_epoch or min(len(loader_a), len(loader_b), len(loader_c))

    weights = {"loss_in": args.in_manifold_loss_factor,
              "loss_out": args.out_of_manifold_loss_factor,
              "loss_out_d": args.discriminative_loss_factor}
    net_loss = NetLoss(weights, args.discriminative_loss_clamp_ratio,
                       args.discriminative_loss_clamp_val, args.loss_p)

    lpips_net = None
    if args.commute_metric == "lpips":
        try:
            import lpips
            lpips_net = lpips.LPIPS(net="alex").to(device)
        except Exception as e:
            print(f"lpips unavailable ({e}); falling back to l1 for the commute term")
            args.commute_metric = "l1"
    D = commute_distance(args.commute_metric, lpips_net)

    params = list(f_a.parameters()) + list(f_b.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2))

    steps_csv = os.path.join(run_dir, "loss_steps.csv")
    with open(steps_csv, "w") as fh:
        fh.write("epoch,step,loss_in_A,loss_out_d_A,loss_in_B,loss_out_d_B,commute,total\n")
    history = []
    fixed_x = None
    iter_a, iter_b, iter_c = infinite(loader_a), infinite(loader_b), infinite(loader_c)

    for epoch in range(args.num_epochs):
        t0 = time.time()
        epoch_sums, epoch_n = {}, 0
        step_rows = []

        for step in range(steps_per_epoch):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            xa = next(iter_a)[0].to(device, non_blocking=True)
            xb = next(iter_b)[0].to(device, non_blocking=True)
            xc = next(iter_c)[0].to(device, non_blocking=True)
            if fixed_x is None:
                fixed_x = xc[:8].clone()

            f_a_copy.load_state_dict(f_a.state_dict())
            f_b_copy.load_state_dict(f_b.state_dict())

            stats_a = freq_stats_from_images(xa) if args.noise_type == "x_fft_stats" else None
            stats_b = freq_stats_from_images(xb) if args.noise_type == "x_fft_stats" else None
            za = make_noise(xa, args.noise_type, stats_a)
            zb = make_noise(xb, args.noise_type, stats_b)

            loss_a, log_a = net_loss(f_a, f_a_copy, xa, za)
            loss_b, log_b = net_loss(f_b, f_b_copy, xb, zb)

            ab = f_a(f_b(xc))
            ba = f_b(f_a(xc))
            commute = D(ab, ba)

            total = loss_a + loss_b + args.lambda_commute * commute

            opt.zero_grad(set_to_none=True)
            total.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            row = {"loss_in_A": log_a["loss_in"], "loss_out_d_A": log_a["loss_out_d"],
                  "loss_in_B": log_b["loss_in"], "loss_out_d_B": log_b["loss_out_d"],
                  "commute": float(commute.detach()), "total": float(total.detach())}
            for k, v in row.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + v
            epoch_n += 1
            step_rows.append(f"{epoch},{step}," + ",".join(f"{row[k]:.6f}" for k in
                             ("loss_in_A", "loss_out_d_A", "loss_in_B", "loss_out_d_B",
                              "commute", "total")) + "\n")
            if step % args.log_every == 0:
                print(f"  e{epoch:03d} s{step:04d} " +
                     " ".join(f"{k}={v:.4f}" for k, v in row.items()))

        means = {k: v / max(epoch_n, 1) for k, v in epoch_sums.items()}
        print(f"[{args.run_name}] epoch {epoch} ({time.time()-t0:.0f}s) " +
             " ".join(f"{k}={v:.4f}" for k, v in means.items()))
        with open(steps_csv, "a") as fh:
            fh.writelines(step_rows)
        history.append({"epoch": epoch, **means})
        with open(os.path.join(run_dir, "loss_history.json"), "w") as fh:
            json.dump(history, fh)
        plot_curves(history, run_dir)

        if epoch % args.sample_every == 0 or epoch == args.num_epochs - 1:
            save_commute_grid(f_a, f_b, fixed_x, run_dir, epoch)

        done = epoch + 1  # epochs completed
        if done % args.save_every == 0 or epoch == args.num_epochs - 1:
            # freq stats travel with the checkpoint (as train/train_ign.py does) so
            # experiments/method1_single_ign/run_experiment.py's --init noise samples matched, on-manifold
            # noise instead of falling back to white noise
            for name, model, arch_name, attr_spec, stats in (
                    ("A", f_a, arch_a, attr_a, stats_a), ("B", f_b, arch_b, attr_b, stats_b)):
                payload = {"epoch": epoch, "epochs_done": done,
                          "model_state_dict": model.state_dict(),
                          "args": vars(args) | {"arch": arch_name},
                          "attr_spec": attr_spec}
                if stats is not None:
                    payload["freq_stats"] = {k: v.detach().cpu() for k, v in stats.items()}
                torch.save(payload, os.path.join(run_dir, f"ckpt_{name}_last.pt"))
                torch.save(payload, os.path.join(run_dir, f"ckpt_{name}_e{done:04d}.pt"))

    print(f"[{args.run_name}] done -> {run_dir}/ckpt_A_last.pt, ckpt_B_last.pt")


if __name__ == "__main__":
    train(get_args_parser().parse_args())
