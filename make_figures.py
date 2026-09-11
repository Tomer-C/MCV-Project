"""Generate report figures from a results.json + tensors.pt produced by run_experiment.py.

  python make_figures.py --results results/Smiling_AND_Eyeglasses
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ign_utils import denorm_for_plot

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def get_args_parser():
    p = argparse.ArgumentParser("Plot alternating-projection results")
    p.add_argument("--results", required=True, help="results/<tag> directory")
    p.add_argument("--dpi", default=150, type=int)
    return p


def fig_convergence(res, fig_dir, dpi):
    """Step norm and per-model idempotency residual vs iteration."""
    alt = res["alternating"]
    iters = np.arange(1, len(alt["step_norm_mean"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iters, alt["step_norm_mean"], "o-", color=PALETTE[0])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel(r"$\|x_{n+1}-x_n\|_2$")
    axes[0].set_title("Step size (Cauchy behaviour)")
    axes[0].grid(alpha=0.3)

    residual_keys = [k for k in alt if k.startswith("idem_residual_")]
    for i, k in enumerate(sorted(residual_keys)):
        axes[1].plot(iters, alt[k], "o-", color=PALETTE[i % len(PALETTE)],
                     label=k.replace("idem_residual_", r"$\|f_{") + r"}(x_n)-x_n\|$")
    axes[1].set_xlabel("iteration $n$")
    axes[1].set_ylabel("idempotency residual")
    axes[1].set_title("Distance to each fixed-point set")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, alt["cycle_distance_mean"], "o-", color=PALETTE[3], label="cycle distance")
    axes[2].plot(iters, alt["image_norm_mean"], "s--", color=PALETTE[2], label=r"$\|x_n\|$")
    axes[2].set_xlabel("iteration $n$")
    axes[2].set_title("Limit cycles / divergence check")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("Alternating projection convergence diagnostics")
    fig.tight_layout()
    path = os.path.join(fig_dir, "convergence.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_damping(res, fig_dir, dpi):
    if "damped" not in res:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    alt = res["alternating"]
    n = np.arange(1, len(alt["step_norm_mean"]) + 1)
    ax.plot(n, alt["step_norm_mean"], "o-", color=PALETTE[0], label="undamped")
    d = res["damped"]
    ax.plot(np.arange(1, len(d["step_norm_mean"]) + 1), d["step_norm_mean"], "s-",
            color=PALETTE[1], label=fr"damped $\alpha$={d['alpha']}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration $n$")
    ax.set_ylabel(r"$\|x_{n+1}-x_n\|_2$")
    ax.set_title("Effect of under-relaxation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "damping.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_attr_rates(res, fig_dir, dpi):
    """Bar chart: joint attribute satisfaction across every method."""
    if "attr_rates" not in res:
        return
    order = ["x0"] + [k for k in res["attr_rates"] if k.startswith("single_")] + \
            ["alternating", "baseline", "real_intersection"]
    keys = [k for k in order if k in res["attr_rates"]]
    labels = {"x0": "$x_0$ (input)", "alternating": "alternating $x^*$",
              "baseline": r"baseline $f_{A\cap B}$", "real_intersection": "real $A\\cap B$"}

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(keys)), 4.5))
    vals = [res["attr_rates"][k]["joint"] for k in keys]
    names = [labels.get(k, k.replace("single_", "") + " alone") for k in keys]
    colors = [PALETTE[2] if k in ("alternating",) else
              PALETTE[3] if k == "baseline" else
              "#999999" if k in ("x0", "real_intersection") else PALETTE[0] for k in keys]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction with ALL target attributes")
    ax.set_title("Joint attribute satisfaction (classifier-judged)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, "attr_rates.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_distribution(res, fig_dir, dpi):
    if "distribution" not in res:
        return
    dist = res["distribution"]
    keys = [k for k in dist if isinstance(dist[k], dict) and "fid" in dist[k]]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["fid", "kid"]):
        vals = [dist[k][metric] for k in keys]
        colors = [PALETTE[2] if k == "alternating" else
                  PALETTE[3] if k == "baseline" else PALETTE[0] for k in keys]
        ax.bar([k.replace("single_", "") for k in keys], vals, color=colors)
        ax.set_title(f"{metric.upper()} vs real $A \\cap B$ ({dist['feature_kind']} features)")
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, "distribution.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_orders(res, fig_dir, dpi):
    if "orders" not in res:
        return
    keys = list(res["orders"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(keys, [res["orders"][k]["joint_residual_final"] for k in keys], color=PALETTE[0])
    axes[0].set_ylabel("joint fixed-point residual")
    axes[0].set_title("Final residual by application order")
    if "joint_attr_rate" in res["orders"][keys[0]]:
        axes[1].bar(keys, [res["orders"][k]["joint_attr_rate"] for k in keys], color=PALETTE[2])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("joint attribute rate")
        axes[1].set_title("Attribute satisfaction by order")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path = os.path.join(fig_dir, "orders.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def fig_trajectory(tensors, fig_dir, dpi, num_rows=6, max_cols=9):
    """Image strip: x_0, x_1, ... x_n for a few samples."""
    traj = tensors["trajectory"]
    cols = min(len(traj), max_cols)
    idx = np.linspace(0, len(traj) - 1, cols).round().astype(int)
    fig, axes = plt.subplots(num_rows, cols, figsize=(1.4 * cols, 1.4 * num_rows))
    for r in range(num_rows):
        for c, t in enumerate(idx):
            ax = axes[r, c] if num_rows > 1 else axes[c]
            ax.imshow(denorm_for_plot(traj[t][r]))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"$x_{{{t}}}$", fontsize=10)
    fig.suptitle("Alternating projection trajectory  $x_{n+1}=(f_B\\circ f_A)(x_n)$")
    fig.tight_layout()
    path = os.path.join(fig_dir, "trajectory.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main(args):
    with open(os.path.join(args.results, "results.json")) as fh:
        res = json.load(fh)
    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"writing figures to {fig_dir}")

    fig_convergence(res, fig_dir, args.dpi)
    fig_damping(res, fig_dir, args.dpi)
    fig_attr_rates(res, fig_dir, args.dpi)
    fig_distribution(res, fig_dir, args.dpi)
    fig_orders(res, fig_dir, args.dpi)

    tpath = os.path.join(args.results, "tensors.pt")
    if os.path.exists(tpath):
        fig_trajectory(torch.load(tpath, map_location="cpu", weights_only=False),
                       fig_dir, args.dpi)

    print_summary(res)


def print_summary(res):
    """A compact text table, convenient to paste into the report."""
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    alt = res["alternating"]
    print(f"converged: {alt['converged']}   diverged: {alt['diverged']}")
    print(f"step norm {alt['step_norm_mean'][0]:.4f} -> {alt['step_norm_mean'][-1]:.4f}")
    print(f"joint fixed-point residual: {alt['joint_residual_final']:.4f}")
    if "attr_rates" in res:
        print("\njoint attribute satisfaction:")
        for k, v in res["attr_rates"].items():
            print(f"  {k:22s} {v['joint']:.3f}")
    if "distribution" in res:
        print(f"\nFID / KID vs real intersection ({res['distribution']['feature_kind']}):")
        for k, v in res["distribution"].items():
            if isinstance(v, dict) and "fid" in v:
                print(f"  {k:22s} FID={v['fid']:8.2f}  KID={v['kid']:.4f}")
    if "paired_vs_baseline" in res:
        print("\nalternating vs baseline (same $x_0$):")
        for k, v in res["paired_vs_baseline"].items():
            if isinstance(v, float):
                print(f"  {k:22s} {v:.4f}")
    if "order_disagreement" in res:
        print("\norder disagreement (L2 between limits):")
        for k, v in res["order_disagreement"].items():
            print(f"  {k}: {v:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
