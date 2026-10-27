"""Method 2: how the number of alternating cycles K affects the output.

Runs x_{k+1} = f_A(f_B(x_k)) for K cycles with two independently trained IGNs
and saves:
  evolution.png    rows = iteration k (k=0 is the starting point), cols = samples
  diagnostics.png  per-iteration convergence diagnostics (batch mean +- std)
  metrics.json     the same diagnostics as numbers

Defaults: A = Smiling (Smiling_paper500 @ epoch 25), B = Male
(Male_500_withsamples @ epoch 75) -- the projector checkpoints used throughout
RESULTS.md. With the default order f_Male is applied first, then f_Smiling
(RESULTS.md law 3: the sharp net generates, the fragile feature goes last).

Two kinds of test, chosen by the starting point x_0 (--init):
  noise  (test 1, default) generation from FFT-matched noise. By default its
         per-frequency statistics come from 256 CelebA train images matching
         --noise_stats_spec ("Male,Smiling", as in RESULTS.md);
         --noise_stats_spec "" uses the statistics stored in the checkpoint of
         the network applied first instead. Output: results/method2_iterations/
  real   (test 2) editing real CelebA test photos. By default these are the
         same faces the other experiments use: CelebA test photos with
         neither Male nor Smiling, shuffled with seed 0 -- exactly the
         selection of real_edit_eval.py, whose first 8 are the faces in
         results/limits/real_image_editing_grid.png and in the
         order-of-application-experiment branch (CelebA files 182920,
         193098, 197965, 200109, 182731, 192184, 192861, 188645). The top row
         is the original photo; each column is labelled with what the
         original shows and its CelebA file name.
         Output: results/method2_real/
  white  plain N(0,1) noise

Usage:
    python test_method2.py                   # test 1: from noise
    python test_method2.py --init real       # test 2: from real photos
    python test_method2.py --init real --real_spec all --real_exclude "Male,Smiling"
    python test_method2.py --num_iters 20 --out_dir results/method2_k20
    python test_method2.py --order BA        # x_{k+1} = f_B(f_A(x_k)) instead

Order comparison (Male first vs Smiling first, same x_0 for both):
    python test_method2.py --compare_orders              # -> results/method2_orders_noise/
    python test_method2.py --compare_orders --init real  # -> results/method2_orders_real/

Notes
-----
* Use early checkpoints (~epoch 25-100): RESULTS.md law 1 -- projection
  strength decays later in training.
* CelebA is expected at <data_root>/celeba/ (see ign_data.py).
* BatchNorm runs in train mode (batch statistics), as in every experiment in
  this repo, so outputs depend on the whole batch. --batch_size controls how
  many images are pushed through together; only --show of them are drawn.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from alternating import alternating_projection
from ign_models import load_ign
from ign_data import (ATTR_TO_IDX, AttrImageDataset, get_attr_indices,
                      get_filtered_dataset)
from ign_utils import freq_stats_from_images, sample_freq_noise, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_a", default="runs/Smiling_paper500/ckpt_0025.pt")
    p.add_argument("--ckpt_b", default="runs/Male_500_withsamples/ckpt_0075.pt")
    p.add_argument("--order", choices=["AB", "BA"], default="AB",
                   help="AB: x_{k+1} = f_A(f_B(x_k)) (proposal's notation); "
                        "BA: x_{k+1} = f_B(f_A(x_k))")
    p.add_argument("--compare_orders", action="store_true",
                   help="run both orders from the same x_0 and save "
                        "order_comparison.png instead of evolution.png")
    p.add_argument("--num_iters", type=int, default=40, help="number of cycles K")
    p.add_argument("--show_iters", default="0,1,2,3,4,6,8,12,20,30,40",
                   help="iterations drawn as rows in evolution.png")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--show", type=int, default=8, help="samples drawn per row")
    p.add_argument("--init", choices=["noise", "real", "white"], default="noise",
                   help="noise: FFT-matched noise (what the IGNs were trained on); "
                        "real: CelebA test photos; white: plain N(0,1)")
    p.add_argument("--noise_stats_spec", default="Male,Smiling",
                   help="CelebA subset whose FFT statistics shape the noise; "
                        "'' = use the checkpoint's stored statistics")
    p.add_argument("--real_spec", default="~Male,~Smiling",
                   help="CelebA subset the real starting photos come from "
                        "('~X' = without X, 'all' = any face). With the default "
                        "and --seed 0 the first 8 are the repo's shared faces")
    p.add_argument("--real_exclude", default="",
                   help="drop photos matching this spec from --real_spec "
                        "('' = drop nothing)")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None,
                   help="default: results/method2_real for --init real, "
                        "results/method2_iterations otherwise")
    p.add_argument("--device", default=None)
    return p.parse_args()


def attr_name(ckpt, path):
    """Human-readable feature name; '~X' / '-X' specs mean 'not X'."""
    spec = ckpt.get("attr_spec") or os.path.basename(os.path.dirname(path))
    return spec.replace("~", "not ").replace("-", "not ")


def random_images(args, spec, split, n, exclude=""):
    """n random CelebA images matching `spec` but not `exclude` (seeded).

    Same selection as metrics.collect_real_images, so with the same spec and
    seed the images match the other experiment scripts.
    Returns (images, attrs, filenames); attrs is (n, 40) with 0/1 entries.
    """
    data_args = argparse.Namespace(data_root=args.data_root, image_size=64)
    ds = get_filtered_dataset(data_args, spec, split=split)
    if exclude:
        drop = set(get_attr_indices(ds.base, exclude))
        ds = AttrImageDataset(ds.base, [i for i in ds.indices if i not in drop])
    if len(ds) < n:
        raise SystemExit(f"only {len(ds)} {split} images match {spec!r} "
                         f"excluding {exclude!r}; need {n}")
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    items = [ds[int(i)] for i in idx]
    names = [ds.base.filename[ds.indices[int(i)]] for i in idx]
    return (torch.stack([x for x, _ in items], 0), torch.stack([a for _, a in items], 0),
            names)


def describe_face(attrs, filename):
    """Column label for a real photo, e.g. 'woman\\nsmiling\\n182920.jpg'."""
    who = "man" if attrs[ATTR_TO_IDX["Male"]] == 1 else "woman"
    smile = "smiling" if attrs[ATTR_TO_IDX["Smiling"]] == 1 else "not smiling"
    return f"{who}\n{smile}\n{filename}"


def starting_point(args, ckpt, ckpt_path, device):
    """Returns (x0, description, column labels or None)."""
    if args.init == "real":
        x0, attrs, names = random_images(args, args.real_spec, "test", args.batch_size,
                                         exclude=args.real_exclude)
        desc = f"real CelebA test photos ({args.real_spec}"
        desc += f", excluding {args.real_exclude}" if args.real_exclude else ""
        desc += f", seed {args.seed})"
        return x0.to(device), desc, [describe_face(a, n) for a, n in zip(attrs, names)]
    if args.init == "noise" and args.noise_stats_spec:
        imgs, _, _ = random_images(args, args.noise_stats_spec, "train", 256)
        stats = freq_stats_from_images(imgs.to(device))
        return (sample_freq_noise(stats, args.batch_size),
                f"noise with FFT stats of {args.noise_stats_spec}", None)
    if args.init == "noise":
        stats = ckpt.get("freq_stats")
        if stats is None:
            raise SystemExit(f"{ckpt_path} carries no freq_stats; use --init white")
        stats = {k: v.to(device) for k, v in stats.items()}
        return (sample_freq_noise(stats, args.batch_size),
                "noise with checkpoint FFT stats", None)
    return torch.randn(args.batch_size, 3, 64, 64, device=device), "white noise", None


def rescale_per_image(x):
    """Min-max each image to [-1, 1]. The starting noise ranges well beyond
    [-1, 1] (about -5..3), so clamping it like the outputs would hide it."""
    lo = x.flatten(1).min(1).values.view(-1, 1, 1, 1)
    hi = x.flatten(1).max(1).values.view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo).clamp_min(1e-8) * 2 - 1


def save_evolution(trajectory, show_iters, n_show, title, path, real_input,
                   col_labels=None):
    rows = [trajectory[k][:n_show].cpu() for k in show_iters]
    # noise ranges well outside [-1, 1] and is rescaled to be visible; real
    # photos are already in [-1, 1] and are shown as they are
    rows = [rescale_per_image(r) if k == 0 and not real_input else r.clamp(-1, 1)
            for k, r in zip(show_iters, rows)]
    grid = make_grid(torch.cat(rows, 0), nrow=n_show, padding=2,
                     normalize=True, value_range=(-1, 1))
    side = trajectory[0].shape[-1]
    fig, ax = plt.subplots(figsize=(1.0 * n_show + 1, 1.0 * len(show_iters) + 0.9))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_yticks([2 + r * (side + 2) + side / 2 for r in range(len(show_iters))])
    k0_label = "k=0 (original)" if real_input else "k=0 (input)"
    ax.set_yticklabels([k0_label if k == 0 else f"k={k}" for k in show_iters])
    if col_labels:
        ax.xaxis.tick_top()
        ax.set_xticks([2 + c * (side + 2) + side / 2 for c in range(n_show)])
        ax.set_xticklabels(col_labels[:n_show], fontsize=7)
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_order_comparison(traj_1, traj_2, order_labels, show_iters, n_show, title,
                          path, real_input, col_labels=None):
    """Both orders side by side: each sample gets two adjacent columns
    (order 1 | order 2), rows = iteration k."""
    rows = []
    for k in show_iters:
        a, b = traj_1[k][:n_show].cpu(), traj_2[k][:n_show].cpu()
        if k == 0 and not real_input:
            a, b = rescale_per_image(a), rescale_per_image(b)
        else:
            a, b = a.clamp(-1, 1), b.clamp(-1, 1)
        rows.append(torch.stack([a, b], 1).flatten(0, 1))   # a0 b0 a1 b1 ...
    grid = make_grid(torch.cat(rows, 0), nrow=2 * n_show, padding=2,
                     normalize=True, value_range=(-1, 1))
    side = traj_1[0].shape[-1]
    cell = side + 2
    fig, ax = plt.subplots(figsize=(2.0 * n_show + 1, 1.0 * len(show_iters) + 1.4))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_yticks([2 + r * cell + side / 2 for r in range(len(show_iters))])
    k0_label = "k=0 (original)" if real_input else "k=0 (input)"
    ax.set_yticklabels([k0_label if k == 0 else f"k={k}" for k in show_iters])
    ax.xaxis.tick_top()
    ax.set_xticks([2 + c * cell + side / 2 for c in range(2 * n_show)])
    ax.set_xticklabels(order_labels * n_show, fontsize=7)
    ax.tick_params(axis="x", length=0)
    for p in range(1, n_show):                 # white gap between sample pairs
        ax.axvline(p * 2 * cell + 1, color="white", lw=4)
    if col_labels:
        for p in range(n_show):
            ax.annotate(col_labels[p].replace("\n", ", ", 1), xy=(p * 2 * cell + cell + 1, 1),
                        xycoords=("data", "axes fraction"), xytext=(0, 24),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, pad=56 if col_labels else 30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics(metrics, names, title, path):
    panels = [
        ("step_norm", r"$\|x_{k+1}-x_k\|_2$  (convergence)"),
        ("idem", r"mean $|f(x_k)-x_k|$  (distance to Fix(f))"),
        ("cycle_distance", r"$\min_{m<k}\|x_k-x_m\|_2$  (limit cycles)"),
        ("image_norm", r"$\|x_k\|_2$  (blow-up)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, label) in zip(axes.flat, panels):
        series = ([(f"idem_residual_{i}", f"f_{n}") for i, n in enumerate(names)]
                  if key == "idem" else [(key, None)])
        for k, lab in series:
            m = metrics[k]                       # (num_iters, B)
            it = torch.arange(1, m.shape[0] + 1)
            mean, std = m.mean(1), m.std(1)
            ax.plot(it, mean, label=lab)
            ax.fill_between(it, mean - std, mean + std, alpha=0.2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("cycle k")
        ax.grid(alpha=0.3)
        if key == "idem":
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_orders(args, f_A, f_B, name_A, name_B, ckpt_B, show_iters, n_show, device):
    """Run both orders from the same x_0 and save order_comparison.png."""
    if args.out_dir is None:
        args.out_dir = ("results/method2_orders_real" if args.init == "real"
                        else "results/method2_orders_noise")
    x0, init_desc, col_labels = starting_point(args, ckpt_B, args.ckpt_b, device)

    # alternating_projection applies the list left to right
    runs = [([f_B, f_A], f"{name_B}\nfirst"), ([f_A, f_B], f"{name_A}\nfirst")]
    trajectories = []
    for models, label in runs:
        print(f"running {args.num_iters} cycles, {label.replace(chr(10), ' ')} "
              f"({init_desc}) ...")
        out = alternating_projection(models, x0, num_iters=args.num_iters,
                                     track_trajectory=True, trajectory_every=1)
        trajectories.append(out["trajectory"])

    os.makedirs(args.out_dir, exist_ok=True)
    title = (f"Order comparison, same x_0 for both.  {name_B} first: "
             f"x_{{k+1}} = f_{name_A}(f_{name_B}(x_k));  {name_A} first: "
             f"x_{{k+1}} = f_{name_B}(f_{name_A}(x_k))\nx_0: {init_desc}")
    path = os.path.join(args.out_dir, "order_comparison.png")
    save_order_comparison(trajectories[0], trajectories[1], [l for _, l in runs],
                          show_iters, n_show, title, path,
                          real_input=args.init == "real", col_labels=col_labels)
    print(f"saved {path}")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    print(f"device: {device}")

    missing = [p for p in (args.ckpt_a, args.ckpt_b) if not os.path.isfile(p)]
    if missing:
        raise SystemExit("missing checkpoint(s):\n  " + "\n  ".join(missing))

    f_A, ckpt_A = load_ign(args.ckpt_a, device)
    f_B, ckpt_B = load_ign(args.ckpt_b, device)
    name_A, name_B = attr_name(ckpt_A, args.ckpt_a), attr_name(ckpt_B, args.ckpt_b)
    print(f"f_A: {args.ckpt_a}  (epoch {ckpt_A.get('epoch')}, feature: {name_A})")
    print(f"f_B: {args.ckpt_b}  (epoch {ckpt_B.get('epoch')}, feature: {name_B})")

    # alternating_projection applies the list left to right, so f_A(f_B(x))
    # is [f_B, f_A]
    if args.order == "AB":
        models, names = [f_B, f_A], [name_B, name_A]
        formula = "x_{k+1} = f_A(f_B(x_k))"
        first_ckpt, first_path = ckpt_B, args.ckpt_b
    else:
        models, names = [f_A, f_B], [name_A, name_B]
        formula = "x_{k+1} = f_B(f_A(x_k))"
        first_ckpt, first_path = ckpt_A, args.ckpt_a

    show_iters = sorted({int(k) for k in args.show_iters.split(",")
                         if int(k) <= args.num_iters})
    n_show = min(args.show, args.batch_size)

    if args.compare_orders:
        compare_orders(args, f_A, f_B, name_A, name_B, ckpt_B, show_iters, n_show,
                       device)
        return

    if args.out_dir is None:
        args.out_dir = ("results/method2_real" if args.init == "real"
                        else "results/method2_iterations")

    x0, init_desc, col_labels = starting_point(args, first_ckpt, first_path, device)
    print(f"running {args.num_iters} cycles of {formula} on {args.batch_size} images "
          f"({init_desc}) ...")
    out = alternating_projection(models, x0, num_iters=args.num_iters,
                                 track_trajectory=True, trajectory_every=1)
    metrics = out["metrics"]

    os.makedirs(args.out_dir, exist_ok=True)
    title = f"{formula}   A = {name_A},  B = {name_B}\nx_0: {init_desc}"
    save_evolution(out["trajectory"], show_iters, n_show, title,
                   os.path.join(args.out_dir, "evolution.png"),
                   real_input=args.init == "real", col_labels=col_labels)
    save_diagnostics(metrics, names, title,
                     os.path.join(args.out_dir, "diagnostics.png"))

    summary = {
        "config": vars(args),
        "feature_A": name_A, "feature_B": name_B, "formula": formula,
        "residual_names": {f"idem_residual_{i}": f"f_{n}" for i, n in enumerate(names)},
        "per_iteration_mean": {k: v.mean(1).tolist() for k, v in metrics.items()},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'k':>3} {'step_norm':>10} " +
          " ".join(f"{'res f_' + n:>16}" for n in names))
    for k in show_iters[1:] if show_iters and show_iters[0] == 0 else show_iters:
        row = [metrics["step_norm"][k - 1].mean().item()] + \
              [metrics[f"idem_residual_{i}"][k - 1].mean().item() for i in range(len(names))]
        print(f"{k:>3} {row[0]:>10.4f} " + " ".join(f"{r:>16.4f}" for r in row[1:]))
    print(f"\nsaved evolution.png, diagnostics.png, metrics.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
