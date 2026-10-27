"""Image grids and metric curves for a commuting-loss sweep.

Reads commuting_sweep_eval.py's output directory (results.json +
grid_tensors.pt) and writes, under <eval_dir>/figures/:

  method_comparison/<run>_e<E>_<domain>.png
      the COMMUTING_LOSS_EXPERIMENT.md layout: x0, baseline f_AnB, naive
      1/2/3 iters, commuting-trained 1/2/3 iters -- same inputs in every row
  sweep_overview_<domain>_<n>iter_e<E>.png
      one row per sweep config at epoch E, next to x0 / baseline / naive
  epoch_progression/<run>_<domain>_<n>iter.png
      one config over fine-tuning length: naive (= epoch 0), then snapshots
  metrics_vs_epoch.png
      joint rate, order distance, sharpness and output diversity against
      fine-tuning epochs

Row labels carry the classifier's joint rate over all x_0 (not just the
columns shown), so every grid reads as images *and* numbers.

  python make_commuting_grids.py --eval_dir exp_results/commuting_sweep/male_smiling \\
      --pair_label "Male AND Smiling"
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from ign_utils import denorm_for_plot

DOMAIN_LABELS = {"real": "in-distribution (real-face init)",
                 "fftnoise": "out-of-distribution (FFT-matched noise)"}

# chart styling: lambda -> color (orange = the lambda=0 control, then an
# ordinal light->dark blue ramp for increasing commute weight), lr -> line
# style, commute metric -> marker; the legend names every config
LAMBDA_COLORS = {0.0: "#eb6834", 0.2: "#86b6ef", 1.0: "#3987e5", 5.0: "#1c5cab",
                 20.0: "#0d366b"}
REF_COLOR, INK, MUTED, GRID = "#52514e", "#0b0b0b", "#8a8984", "#e4e3df"


def fmt_lr(lr):
    m, e = f"{lr:.0e}".split("e")
    return f"{m}e{int(e)}"


def config_label(cfg):
    label = (f"λ={cfg['lambda_commute']:g}, lr={fmt_lr(cfg['lr'])}, "
             f"{(cfg.get('commute_metric') or 'l1').upper()}")
    return label + (f", seed {cfg['seed']}" if cfg.get("seed") else "")


def epochs_str(ep):
    return f"{ep} fine-tune epoch{'' if ep == 1 else 's'}"


def lambda_color(lam):
    if lam in LAMBDA_COLORS:
        return LAMBDA_COLORS[lam]
    keys = sorted(k for k in LAMBDA_COLORS if k > 0)
    return LAMBDA_COLORS[min(keys, key=lambda k: abs(k - lam))]


def marker_style(cfg):
    """Filled circles for L1, hollow squares for LPIPS (same lambda color)."""
    if cfg.get("commute_metric") == "lpips":
        c = lambda_color(cfg["lambda_commute"])
        return dict(marker="s", ms=5, mfc="white", mec=c, mew=1.4)
    return dict(marker="o", ms=4.5, mec="white", mew=1.0)


def draw_grid(rows, labels, title, path, n):
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.6 + 1.2, len(rows) * 1.7),
                             squeeze=False)
    for r, (imgs, label) in enumerate(zip(rows, labels)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(denorm_for_plot(imgs[c].float()))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center",
                              labelpad=4)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def jr(entry, dom, n):
    return entry["domains"][dom]["iters"][n - 1]["joint"]


def main():
    p = argparse.ArgumentParser("commuting sweep figures")
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--out", default=None, help="default: <eval_dir>/figures")
    p.add_argument("--pair_label", default="Male AND Smiling")
    p.add_argument("--n", default=6, type=int, help="image columns")
    p.add_argument("--comparison_epochs", nargs="*", type=int, default=None,
                   help="epochs for method_comparison grids (default: 10 and the final one)")
    p.add_argument("--overview_epochs", nargs="*", type=int, default=None,
                   help="epochs for sweep_overview grids (default: the final one)")
    p.add_argument("--progression_every", default=10, type=int)
    p.add_argument("--progression_epochs", nargs="*", type=int, default=None,
                   help="explicit snapshot epochs for epoch_progression grids "
                        "(overrides --progression_every where present)")
    p.add_argument("--only_runs", nargs="*", default=None,
                   help="restrict method_comparison/epoch_progression grids to these runs")
    p.add_argument("--reference_runs", nargs="*",
                   default=["Male_Smiling_commuting_probe", "Male_Blond_commuting_probe"],
                   help="evaluated runs that get method_comparison grids but stay out of "
                        "the sweep overview/curves (e.g. the original 10-epoch probes)")
    args = p.parse_args()
    out = args.out or os.path.join(args.eval_dir, "figures")
    os.makedirs(out, exist_ok=True)

    results = json.load(open(os.path.join(args.eval_dir, "results.json")))
    grids = torch.load(os.path.join(args.eval_dir, "grid_tensors.pt"))
    ref = results.pop("reference")
    has_base = "baseline" in grids
    naive = results["naive@0"]
    max_iters = len(naive["domains"]["real"]["iters"])

    runs = {}
    for key, e in results.items():
        if e["run"] != "naive":
            runs.setdefault(e["run"], {})[e["epoch"]] = key
    sweep_runs = {r: eps for r, eps in runs.items() if r not in args.reference_runs
                  and results[next(iter(eps.values()))]["config"].get("lambda_commute") is not None}
    order = sorted(sweep_runs, key=lambda r: (
        results[next(iter(sweep_runs[r].values()))]["config"].get("commute_metric") or "",
        -results[next(iter(sweep_runs[r].values()))]["config"]["lr"],
        results[next(iter(sweep_runs[r].values()))]["config"]["lambda_commute"]))

    def head_rows(dom):
        rows, labels = [grids["x0"][dom]], [f"x0 (input)\njoint {ref['x0'][dom]['joint']:.2f}"]
        if has_base:
            rows.append(grids["baseline"][dom])
            labels.append(f"baseline f_A∩B\njoint {ref['baseline'][dom]['joint']:.2f}")
        return rows, labels

    # ---- 1. method comparison, the original layout, one per (run, epoch) --
    for run, eps in runs.items():
        if args.only_runs and run not in args.only_runs:
            continue
        cfg = results[next(iter(eps.values()))]["config"]
        final = max(eps)
        wanted = args.comparison_epochs or sorted({min(10, final), final})
        for ep in [e for e in wanted if e in eps]:
            key = eps[ep]
            for dom in DOMAIN_LABELS:
                rows, labels = head_rows(dom)
                for n in range(1, max_iters + 1):
                    rows.append(grids["naive@0"][dom][n - 1])
                    labels.append(f"naive {n} iter\njoint {jr(naive, dom, n):.2f}")
                for n in range(1, max_iters + 1):
                    rows.append(grids[key][dom][n - 1])
                    labels.append(f"commute {n} iter\njoint {jr(results[key], dom, n):.2f}")
                cl = config_label(cfg) if cfg.get("lambda_commute") is not None else run
                title = (f"{args.pair_label} — {DOMAIN_LABELS[dom]}\n"
                         f"commute-trained: {cl}, {epochs_str(ep)}")
                draw_grid(rows, labels, title,
                          os.path.join(out, "method_comparison", f"{run}_e{ep:03d}_{dom}.png"),
                          args.n)

    # ---- 2. sweep overview: every config side by side --------------------
    if sweep_runs:
        common = set.intersection(*(set(eps) for eps in sweep_runs.values()))
        ov_epochs = args.overview_epochs or ([max(common)] if common else [])
        for ep in ov_epochs:
            for dom in DOMAIN_LABELS:
                for n in sorted({1, max_iters}):
                    rows, labels = head_rows(dom)
                    rows.append(grids["naive@0"][dom][n - 1])
                    labels.append(f"naive (no fine-tune)\njoint {jr(naive, dom, n):.2f}")
                    for run in order:
                        key = sweep_runs[run].get(ep)
                        if key is None:
                            continue
                        rows.append(grids[key][dom][n - 1])
                        labels.append(f"{config_label(results[key]['config'])}\n"
                                      f"joint {jr(results[key], dom, n):.2f}")
                    title = (f"{args.pair_label} — {DOMAIN_LABELS[dom]}\n"
                             f"(f_B∘f_A)^{n}(x0) after {epochs_str(ep)}, one row per config")
                    draw_grid(rows, labels, title,
                              os.path.join(out, f"sweep_overview_{dom}_{n}iter_e{ep:03d}.png"),
                              args.n)

    # ---- 3. epoch progression per config ----------------------------------
    for run, eps in sweep_runs.items():
        cfg = results[next(iter(eps.values()))]["config"]
        if args.only_runs and run not in args.only_runs:
            continue
        shown = [e for e in sorted(eps) if e % args.progression_every == 0] or sorted(eps)
        if args.progression_epochs:
            shown = [e for e in args.progression_epochs if e in eps] or shown
        for dom in DOMAIN_LABELS:
            for n in sorted({1, max_iters}):
                rows, labels = head_rows(dom)
                rows.append(grids["naive@0"][dom][n - 1])
                labels.append(f"epoch 0 (naive)\njoint {jr(naive, dom, n):.2f}")
                for ep in shown:
                    rows.append(grids[eps[ep]][dom][n - 1])
                    labels.append(f"epoch {ep}\njoint {jr(results[eps[ep]], dom, n):.2f}")
                title = (f"{args.pair_label} — {DOMAIN_LABELS[dom]}\n"
                         f"{config_label(cfg)}: (f_B∘f_A)^{n}(x0) vs fine-tune length")
                draw_grid(rows, labels, title,
                          os.path.join(out, "epoch_progression", f"{run}_{dom}_{n}iter.png"),
                          args.n)

    # ---- 4. metric curves ---------------------------------------------------
    if sweep_runs:
        panels = [("joint", 1, "joint rate, 1 iter"),
                  ("joint", max_iters, f"joint rate, {max_iters} iters"),
                  ("order_dist", 1, "order distance, 1 iter"),
                  ("sharpness_rel", max_iters, f"sharpness vs real faces, {max_iters} iters"),
                  ("diversity_rel", max_iters, f"output diversity vs real faces, {max_iters} iters")]
        fig, axes = plt.subplots(2, len(panels), figsize=(4.0 * len(panels), 7.4))
        for r, dom in enumerate(DOMAIN_LABELS):
            for c, (metric, n, title) in enumerate(panels):
                ax = axes[r, c]
                for run in order:
                    eps = sweep_runs[run]
                    cfg = results[next(iter(eps.values()))]["config"]
                    xs = [0] + sorted(eps)
                    ys = [naive["domains"][dom]["iters"][n - 1][metric]] + \
                         [results[eps[e]]["domains"][dom]["iters"][n - 1][metric]
                          for e in sorted(eps)]
                    ax.plot(xs, ys, color=lambda_color(cfg["lambda_commute"]), lw=1.6,
                            ls="-" if cfg["lr"] >= 1e-4 else (0, (4, 2)),
                            solid_capstyle="round", **marker_style(cfg))
                if metric == "joint":
                    if has_base:
                        ax.axhline(ref["baseline"][dom]["joint"], color=REF_COLOR, lw=1,
                                   ls=(0, (1, 2)))
                        ax.annotate("trained f_A∩B", (1, ref["baseline"][dom]["joint"]),
                                    xycoords=("axes fraction", "data"), ha="right",
                                    va="bottom", fontsize=8, color=REF_COLOR)
                    ceil = ref["real_intersection"]["joint"]
                    # the two reference lines can sit ~0.03 apart: label the
                    # ceiling under its line when it is the lower of a close pair
                    below = has_base and 0 <= ref["baseline"][dom]["joint"] - ceil < 0.08
                    ax.axhline(ceil, color=MUTED, lw=1, ls=(0, (1, 2)))
                    ax.annotate("classifier ceiling", (1, ceil),
                                xycoords=("axes fraction", "data"), ha="right",
                                va="top" if below else "bottom", fontsize=8, color=MUTED)
                    ax.set_ylim(0, 1)
                if metric in ("sharpness_rel", "diversity_rel"):
                    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (1, 2)))
                    ax.set_ylim(bottom=0)
                ax.set_title(f"{title}\n{DOMAIN_LABELS[dom]}", fontsize=9, color=INK)
                ax.set_xlabel("fine-tune epochs (0 = naive warm start)", fontsize=8,
                              color=REF_COLOR)
                ax.grid(color=GRID, lw=0.8)
                ax.tick_params(labelsize=8, colors=REF_COLOR)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                for s in ("left", "bottom"):
                    ax.spines[s].set_color(GRID)
        handles = []
        for run in order:
            cfg = results[next(iter(sweep_runs[run].values()))]["config"]
            handles.append(Line2D([], [], color=lambda_color(cfg["lambda_commute"]), lw=1.6,
                                  ls="-" if cfg["lr"] >= 1e-4 else (0, (4, 2)),
                                  label=config_label(cfg), **marker_style(cfg)))
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
                   fontsize=8, frameon=False)
        fig.suptitle(f"{args.pair_label} — commuting fine-tune sweep", fontsize=11)
        fig.tight_layout(rect=[0, 0.07, 1, 0.96])
        fig.savefig(os.path.join(out, "metrics_vs_epoch.png"), dpi=130)
        plt.close(fig)

    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
