"""Train one IGN on an attribute-filtered subset of CelebA.

The loss structure is taken verbatim from IGN-main/main.py (Shocher et al.):

    loss_in    : |f(x) - x|^p            -- x is a fixed point (idempotence on the manifold)
    loss_out   : |f_copy(f(z)) - f(z)|^p -- tighten idempotence, gradient through the INNER f
    loss_out_d : -|f(f(z)) - f(z)|^p     -- push the manifold in, gradient through the OUTER f
                                            (clamped, so it cannot dominate)

The which-f-gets-the-gradient split is the subtle part of IGN and is preserved
exactly: `f_copy` is a detached snapshot updated every step, so `loss_out`
carries gradient only through the inner application, and `ff_z` uses a detached
`f_z` so `loss_out_d` carries gradient only through the outer one.

Usage
-----
  python train_ign.py --attr_spec "Smiling" --run_name f_A
  torchrun --nproc_per_node=4 train_ign.py --attr_spec "Smiling" --ddp --run_name f_A
"""

import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from ign_data import get_data_loader, spec_to_slug
from ign_models import build_ign, unwrap
from ign_utils import (LossManager, ddp_setup, ddp_cleanup, freq_stats_from_images,
                       is_main, sample_freq_noise, save_image_grid, set_seed,
                       update_ema, resolve_device)


def get_args_parser():
    p = argparse.ArgumentParser("Train an IGN on an attribute-filtered CelebA subset")

    # what to train on -- the core addition over upstream
    p.add_argument("--attr_spec", default="Smiling",
                   help="Comma-separated CelebA attributes, '-' to negate. "
                        "e.g. 'Smiling' or 'Smiling,Eyeglasses' or 'Smiling,-Male'. "
                        "Use 'all' for the unfiltered dataset.")
    p.add_argument("--run_name", default=None, help="defaults to a slug of --attr_spec")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--num_workers", default=2, type=int)

    # optimization (paper Appendix C; lr/betas/batch also match run_celeba.sh)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--num_epochs", default=60, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--beta1", default=0.5, type=float)
    p.add_argument("--beta2", default=0.999, type=float)
    p.add_argument("--arch", default="dcgan", choices=["dcgan", "dcgan_enhanced"],
                   help="dcgan = the plain autoencoder the paper describes; "
                        "dcgan_enhanced = the skeleton's residual variant")
    p.add_argument("--norm", default="auto", choices=["auto", "batch", "instance", "none"])
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", default=0.999, type=float)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--grad_clip", action="store_true")
    p.add_argument("--seed", default=0, type=int)

    # IGN loss weights (paper Appendix C values; IGN-main/run_celeba.sh instead
    # used in=1.0, out=1.0, d=10.0 with clamp_ratio=1.0)
    p.add_argument("--in_manifold_loss_factor", default=20, type=float)
    p.add_argument("--out_of_manifold_loss_factor", default=20, type=float)
    p.add_argument("--discriminative_loss_factor", default=2.5, type=float)
    p.add_argument("--discriminative_loss_clamp_ratio", default=1.5, type=float)
    p.add_argument("--discriminative_loss_clamp_val", default=9999.0, type=float)
    p.add_argument("--epoch_begin_idempotence_losses", default=0, type=int)
    p.add_argument("--loss_p", default=1.0, type=float)
    p.add_argument("--noise_type", default="x_fft_stats",
                   choices=["x_fft_stats", "normal", "uniform"],
                   help="x_fft_stats = noise with the batch's per-frequency "
                        "statistics, as the paper uses for CelebA; normal = "
                        "white gaussian (what run_celeba.sh used)")

    # per-epoch feature check: 4 ground-truth subsets w.r.t. the (first)
    # trained attribute -- train/test x has/lacks -- rendered next to the
    # sample grids. Per subset: `check_per_subset` fixed faces (time-lapse
    # comparable across epochs) plus the same number freshly drawn each epoch.
    p.add_argument("--no_feature_check", action="store_true")
    p.add_argument("--check_per_subset", default=10, type=int)
    p.add_argument("--check_applications", default=2, type=int,
                   help="how many times f is applied in the check grids")
    p.add_argument("--check_classifier", default="runs/classifier/classifier.pt",
                   help="attribute classifier for the per-epoch confidence "
                        "curves (test batches); missing file just disables them")

    # infra
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--device", default=None, help="cuda / cpu / cuda:0; auto-detected if unset")
    p.add_argument("--save_every", default=25, type=int, help="epochs")
    p.add_argument("--sample_every", default=1, type=int, help="epochs")
    p.add_argument("--log_every", default=50, type=int, help="steps")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--max_steps_per_epoch", default=0, type=int,
                   help=">0 truncates each epoch; used for CPU smoke tests")
    return p


def make_noise(x, noise_type, freq_stats=None):
    """Sample z matching x's shape. For x_fft_stats, `freq_stats` are the
    per-frequency statistics of the current batch (paper-style colored noise)."""
    if noise_type == "uniform":
        return torch.rand_like(x) * 2 - 1
    if noise_type == "x_fft_stats":
        return sample_freq_noise(freq_stats, x.shape[0])
    return torch.randn_like(x)


CONF_KEYS = [("clf_p_after_lacks", "P(feat) after f, lacks batch"),
             ("clf_dp_has", "dP has (after - before)"),
             ("clf_dp_lacks", "dP lacks (after - before)")]


def plot_losses(history, run_dir):
    """Loss curves from the per-epoch means: the three raw components, the
    lambda-weighted total, held-out test reconstruction, and (when a
    classifier is available) the test-batch confidence curves. Positive
    series get a log y-axis so the late slow decay stays visible. Skipped
    silently if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def series(k):
        return [float("nan") if h.get(k) is None else h[k] for h in history]

    keys = [("loss_in", "loss_in (reconstruction, raw)"),
            ("loss_out", "loss_out (idempotence, raw)"),
            ("loss_out_d", "loss_out_d (tightness, raw)"),
            ("total", "total (weighted with lambdas)")]
    if any("test_recon_l1" in h for h in history):
        keys.append(("test_recon_l1", "test recon |f(x)-x| (has-only, held-out)"))
    has_conf = any(any(k in h for k, _ in CONF_KEYS) for h in history)

    xs = [h["epoch"] for h in history]
    n_panels = len(keys) + (1 if has_conf else 0)
    ncols, nrows = 2, (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows))
    axes = list(axes.flat)
    for ax, (k, title) in zip(axes, keys):
        ys = series(k)
        ax.plot(xs, ys, marker=".", markersize=2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        vals = [v for v in ys if v == v]  # drop nan
        if vals and min(vals) > 0:
            ax.set_yscale("log")
    if has_conf:
        ax = axes[len(keys)]
        for k, label in CONF_KEYS:
            ax.plot(xs, series(k), label=label, marker=".", markersize=3)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("classifier confidence (test batches)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=120)
    plt.close(fig)


@torch.no_grad()
def confidence_metrics(model, check, n=64):
    """Classifier P(feature) before/after f on FRESH has/lacks test batches.

    A new random batch per call (per epoch), test split only. Returns:
      clf_p_after_lacks  mean P(feature | f(x)) on lacks images -- did f add it?
      clf_dp_has         mean P-after minus P-before on has images -- does f
                         make the classifier LESS sure about real feature-havers?
      clf_dp_lacks       same difference on lacks images.
    Train-mode BN with buffers snapshotted/restored: measurement only.
    """
    s = check["splits"]["test"]
    clf, fi, device = check["clf"], check["feature_idx"], check["device"]
    batches = {}
    for name in ("has", "lacks"):
        pool = s["pools"][name]
        sel = pool[torch.randperm(len(pool), generator=check["rng"])[:n]]
        batches[name] = torch.stack([s["ds"][i][0] for i in sel.tolist()]).to(device)

    buffers = {k: v.clone() for k, v in model.named_buffers()}
    was_training = model.training
    model.train()
    fx = {name: model(x) for name, x in batches.items()}
    for k, v in model.named_buffers():
        v.copy_(buffers[k])
    if not was_training:
        model.eval()

    def p(imgs):
        return torch.sigmoid(clf(imgs))[:, fi]

    dp_has = p(fx["has"]) - p(batches["has"])
    p_lacks_after = p(fx["lacks"])
    dp_lacks = p_lacks_after - p(batches["lacks"])
    return {"clf_p_after_lacks": float(p_lacks_after.mean()),
            "clf_dp_has": float(dp_has.mean()),
            "clf_dp_lacks": float(dp_lacks.mean())}


@torch.no_grad()
def held_out_recon_l1(model, x):
    """Mean |f(x)-x| on a fixed held-out test batch. Train-mode BN with
    buffers snapshotted/restored: pure measurement, no leakage into training."""
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    was_training = model.training
    model.train()
    l1 = float((model(x) - x).abs().mean())
    for k, v in model.named_buffers():
        v.copy_(buffers[k])
    if not was_training:
        model.eval()
    return l1


def setup_feature_check(args, device, run_dir):
    """Prepare the 4 ground-truth subsets (train/test x has/lacks) once.

    Uses saved attr_splits/ id lists when present (else the dataset's own
    labels) for the first attribute of --attr_spec. Per subset,
    `check_per_subset` fixed faces are decoded to GPU now and reused every
    epoch (a comparable time-lapse); the index pools are kept so a fresh
    random draw can be made each epoch. All selected ids are recorded to
    feature_check/ids.json. Also holds a fixed held-out test batch for the
    per-epoch reconstruction curve.
    """
    if args.no_feature_check:
        return None
    import types
    from ign_data import CELEBA_ATTRS, get_filtered_dataset, parse_attr_spec

    feature_idx, _want = parse_attr_spec(args.attr_spec)[0]
    k = args.check_per_subset
    g = torch.Generator().manual_seed(args.seed + 1234)

    check = {"k": k, "apps": args.check_applications, "device": device,
             "rng": g, "splits": {}, "run_dir": run_dir,
             "feature_idx": feature_idx, "clf": None,
             "ids": {"fixed": {}, "fresh": {}}}
    if args.check_classifier and os.path.exists(args.check_classifier):
        from ign_models import load_classifier
        check["clf"], _ = load_classifier(args.check_classifier, device)
        print(f"feature check: classifier {args.check_classifier} loaded -- "
              "confidence curves on")
    else:
        print(f"feature check: no classifier at {args.check_classifier!r} -- "
              "confidence curves off")
    feature_name = CELEBA_ATTRS[feature_idx]
    for split in ("train", "test"):
        ns = types.SimpleNamespace(data_root=args.data_root, image_size=args.image_size)
        ds = get_filtered_dataset(ns, "all", split=split)

        # prefer the id lists written by make_attr_splits.py (project root),
        # fall back to computing the same thing from the dataset labels
        pools, list_dir = None, None
        for root in ("attr_splits", "attr_split"):
            d = os.path.join(root, feature_name, split)
            if (os.path.exists(os.path.join(d, "has.txt"))
                    and os.path.exists(os.path.join(d, "lacks.txt"))):
                list_dir = d
                break
        if list_dir is not None:
            pools = {}
            for name in ("has", "lacks"):
                with open(os.path.join(list_dir, f"{name}.txt")) as fh:
                    pools[name] = torch.tensor(
                        [int(line.split("\t")[0]) for line in fh if line.strip()])
            print(f"feature check [{split}]: using saved id lists from {list_dir} "
                  f"({len(pools['has'])} has / {len(pools['lacks'])} lacks)")
        else:
            base_attr = getattr(getattr(ds, "base", None), "attr", None)
            if base_attr is None:
                # e.g. the synthetic dataset of scripts/integration_test.py
                print("feature check off: dataset exposes no ground-truth "
                      "attribute matrix and no attr_splits/ lists were found")
                return None
            col = base_attr[:, feature_idx]
            pools = {"has": (col == 1).nonzero(as_tuple=True)[0],
                     "lacks": (col == 0).nonzero(as_tuple=True)[0]}
            print(f"feature check [{split}]: no saved lists found "
                  f"(attr_splits/{feature_name}/{split}/) -- computed from labels")
        fixed = []
        check["ids"]["fixed"][split] = {}
        for name in ("has", "lacks"):
            sel = pools[name][torch.randperm(len(pools[name]), generator=g)[:k]]
            check["ids"]["fixed"][split][name] = sel.tolist()
            fixed.append(torch.stack([ds[i][0] for i in sel.tolist()]))
        check["splits"][split] = {"ds": ds, "pools": pools,
                                  "fixed": torch.cat(fixed).to(device)}
        if split == "test":
            # fixed held-out batch for the per-epoch reconstruction curve:
            # has-feature test images only, so |f(x)-x| measures pure
            # reconstruction (on lacks-images f is *supposed* to change x)
            pool = pools["has"]
            sel = pool[torch.randperm(len(pool), generator=g)[:64]]
            check["ids"]["held_out_recon"] = sel.tolist()
            check["recon_x"] = torch.stack(
                [ds[i][0] for i in sel.tolist()]).to(device)

    os.makedirs(os.path.join(run_dir, "feature_check"), exist_ok=True)
    with open(os.path.join(run_dir, "feature_check", "ids.json"), "w") as fh:
        json.dump(check["ids"], fh, indent=2)
    print(f"feature check on: {CELEBA_ATTRS[feature_idx]!r} -- per split/epoch: "
          f"{k} fixed + {k} fresh, each for has & lacks; ids -> feature_check/ids.json")
    return check


@torch.no_grad()
def save_feature_check(model, check, run_dir, epoch):
    """Render the 4-subset grids (fixed + fresh) with train-mode BatchNorm.

    Pure inference: no gradients, no optimizer step, and BN buffers are
    snapshotted/restored -- nothing seen here (test images included) can
    influence training. Fixed and fresh images are forwarded as ONE batch per
    split so they share the same BN statistics.

    Writes per split:  feature_check/<split>_fixed_eNNNN.png
                       feature_check/<split>_fresh_eNNNN.png
    Grid: columns 1..k have the feature, k+1..2k lack it;
          rows are x, f(x), ..., f^apps(x).
    """
    buffers = {name: v.clone() for name, v in model.named_buffers()}
    was_training = model.training
    model.train()
    k = check["k"]
    check["ids"]["fresh"][str(epoch)] = {}
    for split, s in check["splits"].items():
        fresh = []
        check["ids"]["fresh"][str(epoch)][split] = {}
        for name in ("has", "lacks"):
            pool = s["pools"][name]
            sel = pool[torch.randperm(len(pool), generator=check["rng"])[:k]]
            check["ids"]["fresh"][str(epoch)][split][name] = sel.tolist()
            fresh.append(torch.stack([s["ds"][i][0] for i in sel.tolist()]))
        fresh = torch.cat(fresh).to(check["device"])

        x = torch.cat([s["fixed"], fresh])
        rows = {"fixed": [x[:2 * k]], "fresh": [x[2 * k:]]}
        cur = x
        for _ in range(check["apps"]):
            cur = model(cur)
            rows["fixed"].append(cur[:2 * k])
            rows["fresh"].append(cur[2 * k:])
        for kind, r in rows.items():
            save_image_grid(torch.cat(r, 0),
                            os.path.join(run_dir, "feature_check",
                                         f"{split}_{kind}_e{epoch:04d}.png"),
                            nrow=2 * k)
    for name, v in model.named_buffers():
        v.copy_(buffers[name])
    if not was_training:
        model.eval()
    with open(os.path.join(run_dir, "feature_check", "ids.json"), "w") as fh:
        json.dump(check["ids"], fh, indent=2)


def train(args):
    rank = ddp_setup() if args.ddp else 0
    device = resolve_device(args, rank)
    set_seed(args.seed + rank)

    if args.run_name is None:
        args.run_name = spec_to_slug(args.attr_spec)
    run_dir = os.path.join(args.out_dir, args.run_name)
    if is_main():
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        with open(os.path.join(run_dir, "args.json"), "w") as fh:
            json.dump(vars(args), fh, indent=2)
        print(f"[{args.run_name}] device={device} spec={args.attr_spec!r}")

    loader, dataset = get_data_loader(args, args.attr_spec, split="train")
    if is_main():
        print(f"[{args.run_name}] training images: {len(dataset)}")

    f, f_copy = build_ign(args, device)
    lm = LossManager(f, args, run_dir)
    start_epoch = lm.maybe_resume(f, args.resume, device)

    check = setup_feature_check(args, device, run_dir) if is_main() else None

    # loss bookkeeping: raw per-step CSV + per-epoch means JSON + curves PNG
    steps_csv = os.path.join(run_dir, "loss_steps.csv")
    hist_path = os.path.join(run_dir, "loss_history.json")
    loss_history = []
    if is_main():
        if args.resume and os.path.exists(hist_path):
            with open(hist_path) as fh:
                loss_history = json.load(fh)
        if not (args.resume and os.path.exists(steps_csv)):
            with open(steps_csv, "w") as fh:
                fh.write("epoch,step,loss_in,loss_out,loss_out_d,total\n")

    wandb_run = None
    if args.use_wandb and is_main():
        try:
            import wandb
            wandb_run = wandb.init(project="IGN-AltProj", name=args.run_name, config=vars(args))
        except Exception as e:
            print(f"wandb disabled: {e}")

    # A fixed noise batch so sample grids are comparable across epochs. For
    # x_fft_stats it needs data statistics, so it is built from the first batch.
    fixed_z = None
    if args.noise_type != "x_fft_stats":
        fixed_z = make_noise(
            torch.empty(8, 3, args.image_size, args.image_size, device=device),
            args.noise_type)
    freq_stats = None

    for epoch in range(start_epoch, args.num_epochs):
        if args.ddp and hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        t0 = time.time()
        step_rows = []

        for step, batch in enumerate(loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            x = batch[0].to(device, non_blocking=True)
            if args.noise_type == "x_fft_stats":
                freq_stats = freq_stats_from_images(x)
            z = make_noise(x, args.noise_type, freq_stats)
            if fixed_z is None:
                fixed_z = sample_freq_noise(freq_stats, 8)

            # f_copy tracks f: EMA if requested, else an exact snapshot.
            if args.ema:
                update_ema(unwrap(f_copy), unwrap(f), args.ema_decay)
            else:
                unwrap(f_copy).load_state_dict(unwrap(f).state_dict())

            with torch.autocast(device.type, enabled=args.use_amp):
                fx = f(x)
                fz = f(z)
                f_z = fz.detach()          # detached target: kills grad to the inner f
                f_fz = f_copy(fz)          # grad flows through the INNER f only
                ff_z = f(f_z)              # grad flows through the OUTER f only

                lm.add("loss_in", (fx - x).abs().pow(args.loss_p).mean())
                lm.add("loss_out", (f_fz - fz).abs().pow(args.loss_p).mean())
                lm.add("loss_out_d",
                       -lm.clamp_d((ff_z - f_z).abs().pow(args.loss_p)).mean())

            lm.step(epoch)

            if is_main():
                L = lm._last
                step_rows.append(
                    f"{epoch},{step},{L.get('loss_in', float('nan')):.6f},"
                    f"{L.get('loss_out', float('nan')):.6f},"
                    f"{L.get('loss_out_d', float('nan')):.6f},"
                    f"{L.get('total', float('nan')):.6f}\n")
                if step % args.log_every == 0:
                    print(f"  e{epoch:03d} s{step:04d} {lm.last_str()}")

        if is_main():
            msg = f"[{args.run_name}] epoch {epoch} ({time.time()-t0:.0f}s) {lm.epoch_str()}"
            print(msg)
            if wandb_run is not None:
                wandb_run.log(lm.epoch_means() | {"epoch": epoch})

            # persist losses: append raw steps, rewrite epoch means, redraw curves
            with open(steps_csv, "a") as fh:
                fh.writelines(step_rows)
            entry = {"epoch": epoch, **lm.epoch_means()}
            if check is not None:
                entry["test_recon_l1"] = held_out_recon_l1(unwrap(f), check["recon_x"])
                if check["clf"] is not None:
                    entry.update(confidence_metrics(unwrap(f), check))
            loss_history.append(entry)
            with open(hist_path, "w") as fh:
                json.dump(loss_history, fh)
            plot_losses(loss_history, run_dir)

            if epoch % args.sample_every == 0 or epoch == args.num_epochs - 1:
                fresh_z = make_noise(x[:fixed_z.shape[0]], args.noise_type, freq_stats)
                save_samples(unwrap(f), fixed_z, x, run_dir, epoch, fresh_z=fresh_z)
                if check is not None:
                    save_feature_check(unwrap(f), check, run_dir, epoch)
            # Freq stats travel with the checkpoint so experiment scripts can
            # sample matched noise without reloading the dataset.
            extra = None
            if freq_stats is not None:
                extra = {"freq_stats": {k: v.detach().cpu() for k, v in freq_stats.items()}}
            if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
                lm.save_ckpt(unwrap(f), epoch, args, os.path.join(run_dir, "ckpt_last.pt"),
                             extra=extra)
                lm.save_ckpt(unwrap(f), epoch, args,
                             os.path.join(run_dir, f"ckpt_{epoch:04d}.pt"), extra=extra)
        lm.reset_epoch()

    if is_main():
        extra = None
        if freq_stats is not None:
            extra = {"freq_stats": {k: v.detach().cpu() for k, v in freq_stats.items()}}
        lm.save_ckpt(unwrap(f), args.num_epochs - 1, args,
                     os.path.join(run_dir, "ckpt_last.pt"), extra=extra)
        print(f"[{args.run_name}] done -> {run_dir}/ckpt_last.pt")
    ddp_cleanup(args.ddp)


@torch.no_grad()
def save_samples(model, fixed_z, x, run_dir, epoch, fresh_z=None):
    """Grid (top to bottom): fixed z -> f(z) -> f^2(z) -> f^3(z); then, if
    given, a FRESH noise batch (resampled every epoch) with the same three
    applications; then real x and its reconstruction f(x). With the default
    --noise_type x_fft_stats both noise batches are FFT frequency-matched.

    Rendered with train-mode BatchNorm (batch statistics), which is how
    upstream's test() renders all its figures: an IGN's *running* stats are a
    blend of real-image, noise and generated batches, so eval-mode grids can
    look collapsed even when the model is healthy. Buffers are snapshotted and
    restored so these extra forwards do not perturb the running stats.
    """
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    was_training = model.training
    model.train()
    rows = []
    for z in [fixed_z] + ([fresh_z] if fresh_z is not None else []):
        rows.append(z.clamp(-1, 1))
        cur = z
        for _ in range(3):
            cur = model(cur)
            rows.append(cur)
    rows.append(x[:fixed_z.shape[0]])
    rows.append(model(x[:fixed_z.shape[0]]))
    save_image_grid(torch.cat(rows, 0), os.path.join(run_dir, "samples", f"e{epoch:04d}.png"),
                    nrow=fixed_z.shape[0])
    for k, v in model.named_buffers():
        v.copy_(buffers[k])
    if not was_training:
        model.eval()


if __name__ == "__main__":
    train(get_args_parser().parse_args())
