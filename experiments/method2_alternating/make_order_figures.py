"""Figures for experiments/method2_alternating/order_experiment.py.

Per (pair, init):
  <tag>_grid.png   labeled image grid -- every column is one x_0, every row a
                   method/order/iteration, so the two orders can be compared
                   face by face. A |A->B - B->A| heat-map row shows where the
                   orders disagree.
  <tag>_rates.png  joint and per-attribute rates vs iteration for both orders
                   (mean +- std over seeds), with the single nets and baseline.
  <tag>_dist.png   distance between the two orders vs iteration, against the
                   BatchNorm noise floor.

  python -m experiments.method2_alternating.make_order_figures --res exp_results/order_experiment
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ign.utils import denorm_for_plot  # noqa: E402

C_AB, C_BA = "#2563eb", "#dc2626"


def grid(samples, path, n_cols=8, k_show=3):
    A, B = samples["names"]
    rows = [("x$_0$ (start)", samples["x0"]),
            (f"$f_{{{A}}}$ only", samples["half_ab"]),
            (f"$f_{{{B}}}$ only", samples["half_ba"])]
    for k in range(min(k_show, len(samples["ab"]))):
        rows.append((f"{A} → {B}\n{k + 1} cycle{'s' if k else ''}", samples["ab"][k]))
        rows.append((f"{B} → {A}\n{k + 1} cycle{'s' if k else ''}", samples["ba"][k]))
        if k == 0:
            rows.append(("|difference|\n1 cycle", (samples["ab"][0] - samples["ba"][0]).abs()))
    if samples["baseline"] is not None:
        rows.append((f"trained baseline\n$f_{{{A}\\cap {B}}}$", samples["baseline"]))

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.25 * n_cols + 1.9, 1.3 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if label.startswith("|difference|"):
                ax.imshow(imgs[c].mean(0).numpy(), cmap="magma", vmin=0, vmax=1)
            else:
                ax.imshow(denorm_for_plot(imgs[c]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if "→" in label:
                color = C_AB if label.startswith(A) else C_BA
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color(color); s.set_linewidth(2.5)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    for c in range(n_cols):
        axes[0, c].set_title(f"sample {c + 1}", fontsize=8)
    init = samples.get("init_desc", samples["init"])
    fig.suptitle(f"Order of application: {A} + {B}, start = {init}\n"
                 f"blue frame = {A} first ({A} → {B}),  red frame = {B} first ({B} → {A}),  "
                 "same x$_0$ in every column", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rate_plot(entry, path):
    A, B = entry["pair"]
    s = entry["summary"]
    ks = [it["k"] for it in s["iters"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, metric, title in zip(axes, ("joint", A, B),
                                 (f"joint ({A} AND {B})", f"{A} rate", f"{B} rate")):
        for order, color, lab in (("A->B", C_AB, f"{A} → {B}"), ("B->A", C_BA, f"{B} → {A}")):
            m = [it[order][metric]["mean"] for it in s["iters"]]
            sd = [it[order][metric]["std"] for it in s["iters"]]
            ax.errorbar(ks, m, yerr=sd, color=color, marker="o", capsize=3, label=lab)
        ax.axhline(s["x0"][metric]["mean"], color="#9ca3af", ls=":", label="x$_0$")
        ax.axhline(s["single"][A][metric]["mean"], color=C_AB, ls="--", alpha=.5, label=f"$f_{{{A}}}$ only")
        ax.axhline(s["single"][B][metric]["mean"], color=C_BA, ls="--", alpha=.5, label=f"$f_{{{B}}}$ only")
        if "baseline" in s:
            ax.axhline(s["baseline"][metric]["mean"], color="#111827", ls="-.", label="trained baseline")
        ax.set_title(title); ax.set_xlabel("alternation cycles"); ax.set_xticks(ks)
        ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    axes[0].set_ylabel("classifier rate")
    axes[-1].legend(fontsize=8, loc="best")
    init = entry.get("init_desc", entry["init"])
    fig.suptitle(f"{A} + {B}, start = {init}  (mean ± std over {len(entry['seeds'])} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dist_plot(entry, path):
    A, B = entry["pair"]
    s = entry["summary"]
    ks = [it["k"] for it in s["iters"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, m, title in ((axes[0], "l2_mean", "L2 distance (lower = closer)"),
                         (axes[1], "ssim_mean", "SSIM (higher = closer)")):
        for key, color, lab in (("order_distance", "#7c3aed", f"{A} → {B}  vs  {B} → {A}"),
                                ("bn_noise_floor", "#9ca3af", "same order, reshuffled batches (noise floor)")):
            ax.errorbar(ks, [it[key][m]["mean"] for it in s["iters"]],
                        yerr=[it[key][m]["std"] for it in s["iters"]],
                        color=color, marker="o", capsize=3, label=lab)
        ax.set_title(title); ax.set_xlabel("alternation cycles"); ax.set_xticks(ks); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"How far apart do the two orders land?  {A} + {B}, start = {entry.get('init_desc', entry['init'])}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def compare_plot(runs, path, tag="Male+Smiling__noise"):
    """Joint rate vs cycles for both orders, one panel per checkpoint set."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4.3 * len(runs), 3.8), sharey=True)
    for ax, (label, res) in zip(axes, runs):
        entry = json.load(open(os.path.join(res, "results.json")))[tag]
        A, B = entry["pair"]
        s = entry["summary"]
        ks = [it["k"] for it in s["iters"]]
        for order, color, lab in (("A->B", C_AB, f"{A} → {B}"), ("B->A", C_BA, f"{B} → {A}")):
            ax.errorbar(ks, [it[order]["joint"]["mean"] for it in s["iters"]],
                        yerr=[it[order]["joint"]["std"] for it in s["iters"]],
                        color=color, marker="o", capsize=3, label=lab)
        ax.axhline(s["baseline"]["joint"]["mean"], color="#111827", ls="-.", label="trained baseline")
        ax.set_title(label); ax.set_xlabel("alternation cycles"); ax.set_xticks(ks)
        ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    axes[0].set_ylabel(f"joint rate ({A} AND {B})")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Order of application across checkpoints (FFT-matched noise, mean ± std over 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", default=os.path.join("exp_results", "order_experiment"))
    p.add_argument("--compare", nargs="+", default=None,
                   help="label=results_dir pairs: draw one joint-rate panel per checkpoint set")
    p.add_argument("--compare_out", default=os.path.join("exp_results", "order_checkpoint_comparison.png"))
    args = p.parse_args()
    if args.compare:
        compare_plot([s.split("=", 1) for s in args.compare], args.compare_out)
        print("wrote", args.compare_out)
        return
    results = json.load(open(os.path.join(args.res, "results.json")))
    for tag, entry in results.items():
        if not isinstance(entry, dict) or "summary" not in entry:
            continue
        samples = torch.load(os.path.join(args.res, f"{tag}_samples.pt"), weights_only=False)
        grid(samples, os.path.join(args.res, f"{tag}_grid.png"))
        rate_plot(entry, os.path.join(args.res, f"{tag}_rates.png"))
        dist_plot(entry, os.path.join(args.res, f"{tag}_dist.png"))
        print("figures:", tag)


if __name__ == "__main__":
    main()
