"""Does alternating projection extend beyond two classes?

The two-class scheme x_{n+1} = f_B(f_A(x_n)) is generalised to K >= 2
single-feature IGNs applied in a fixed cyclic order,

    x_{n+1} = f_K(...f_2(f_1(x_n))...),

which is the classical cyclic-projection extension of von Neumann's theorem
(Halperin) for subspaces. Nothing guarantees it for IGNs; this script measures
what happens, with plain (unrelaxed) steps -- experiments/method2_alternating/triple_eval.py is the relaxed,
FID-scored variant.

For every seed the SAME x_0 batch is pushed through

  * each single net,            f_i(x_0)
  * the K-way alternation,      k = 1..max_iters cycles
  * the 2-way alternation on the first two features, as the reference the
    K-way result is compared against: does adding more classes keep the
    first two features, or trade them away?

and at every state we record per-attribute rates, the all-K joint rate, every
pairwise joint rate, and the joint fixed-point residual max_i ||f_i(x) - x||
(near zero = in every target set at once).

With --all_orders, every permutation of the features is also run from the
same x_0, and each is compared to the given order with an exact McNemar test
on the per-image all-K joint outcome (as experiments/method2_alternating/order_experiment.py does for pairs).
A feature written ~Attr targets NOT Attr (e.g. ~Young for Young_500).

Outputs (in --out): results.json, samples.pt, grid.png (labeled faces),
rates.png (rates and residual vs cycles, mean +- std over seeds), and with
--all_orders orders_grid.png / orders_rates.png.

Example
-------
  python -m experiments.method2_alternating.multi_class_experiment --runs_dir runs/runs \
      --features Male Smiling Black_Hair --num_samples 512 --seeds 0 1 2
"""

import argparse
import itertools
import json
import os
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ign.alternating import set_inference_mode  # noqa: E402
from ign.data import ATTR_TO_IDX, _celeba_files_present, get_filtered_dataset  # noqa: E402
from ign.models import load_classifier, load_ign  # noqa: E402
from ign.utils import denorm_for_plot, sample_freq_noise, set_seed  # noqa: E402
from ign.metrics import attr_probabilities, collect_real_images  # noqa: E402
from experiments.method2_alternating.order_experiment import CKPTS, apply, mcnemar  # noqa: E402

# Same projector checkpoints as experiments/method2_alternating/order_experiment.py / experiments/method2_alternating/triple_eval.py /
# experiments/configs/*_pairs_cfg.json. A leading "~" means the net projects onto NOT-attribute:
# Young_500 was trained on "~Young" and makes faces older.
CKPTS = dict(CKPTS, Black_Hair="Black_Hair_200/ckpt_0150.pt",
             Eyeglasses="Eyeglasses_200/ckpt_0025.pt", **{"~Young": "Young_500/ckpt_0075.pt"})
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d"]


def get_args_parser():
    p = argparse.ArgumentParser("Alternating projection with K >= 2 classes")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--classifier", default=None,
                   help="defaults to <runs_dir>/classifier/classifier.pt")
    p.add_argument("--features", nargs="+", default=["Male", "Smiling", "Black_Hair"],
                   help="applied in this order within each cycle; ~Attr = NOT Attr")
    p.add_argument("--ckpt", nargs="+", default=[],
                   help="override projector checkpoints, e.g. Black_Hair=Black_Hair_200/ckpt_0025.pt")
    p.add_argument("--init", default="noise", choices=["noise", "real", "none"],
                   help="noise = FFT-matched noise; real = any CelebA test photo; "
                        "none = test photos with none of the features")
    p.add_argument("--noise_stats", default="Male_500_withsamples/ckpt_0075.pt",
                   help="checkpoint under runs_dir whose FFT stats generate the noise x_0 "
                        "(fixed, so x_0 is identical across feature sets and orders); "
                        "'ckpt' = the first net's checkpoint")
    p.add_argument("--all_orders", action="store_true",
                   help="also run every permutation of --features from the same x_0, with an "
                        "exact McNemar test of each order against the given one")
    p.add_argument("--num_samples", default=512, type=int)
    p.add_argument("--seeds", nargs="+", default=[0, 1, 2], type=int)
    p.add_argument("--max_iters", default=4, type=int)
    p.add_argument("--batch_size", default=256, type=int, help="BatchNorm batch size")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--out", default=os.path.join("exp_results", "multi_class"))
    return p


@torch.no_grad()
def run_cycles(nets, x0, k_max, bs):
    """Iterates after every full cycle over `nets`."""
    xs, x = [], x0
    for _ in range(k_max):
        for f in nets:
            x = apply(f, x, bs)
        xs.append(x)
    return xs


@torch.no_grad()
def joint_residual(nets, x, bs):
    """Per-image max_i mean|f_i(x) - x| (batched, so BatchNorm sees `bs`)."""
    r = torch.stack([(apply(f, x, bs) - x).abs().flatten(1).mean(1) for f in nets])
    return r.max(0).values


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    feats = args.features
    assert len(feats) >= 2, "need at least two features"
    CKPTS.update(dict(s.split("=", 1) for s in args.ckpt))
    have_data = _celeba_files_present(args.data_root)
    if args.init != "noise" and not have_data:
        raise SystemExit(f"--init {args.init} needs CelebA under {args.data_root}/celeba")

    clf, _ = load_classifier(args.classifier or os.path.join(args.runs_dir, "classifier",
                                                              "classifier.pt"), device)
    nets, ckpts = [], []
    for name in feats:
        f, ck = load_ign(os.path.join(args.runs_dir, CKPTS[name]), device)
        nets.append(f)
        ckpts.append(ck)
    set_inference_mode(nets, eval_mode=False)
    idx = [ATTR_TO_IDX[a.lstrip("~")] for a in feats]
    neg = torch.tensor([a.startswith("~") for a in feats])
    pairs = list(itertools.combinations(range(len(feats)), 2))

    def measure(x, residual=True):
        """Rates dict, plus the per-image all-K joint outcome (for McNemar)."""
        hit = (attr_probabilities(clf, x, device)[:, idx] > 0.5) ^ neg
        m = {a: hit[:, i].float().mean().item() for i, a in enumerate(feats)}
        m["joint_all"] = hit.all(1).float().mean().item()
        for i, j in pairs:
            m[f"{feats[i]}&{feats[j]}"] = (hit[:, i] & hit[:, j]).float().mean().item()
        if residual:
            m["joint_residual"] = joint_residual(nets, x, args.batch_size).mean().item()
        return m, hit.all(1)

    per_seed = []
    for seed in args.seeds:
        set_seed(seed)
        if args.init == "noise":
            src = ckpts[0] if args.noise_stats == "ckpt" else torch.load(
                os.path.join(args.runs_dir, args.noise_stats), map_location=device, weights_only=False)
            stats = {k: v.to(device) for k, v in src["freq_stats"].items()}
            x0 = sample_freq_noise(stats, args.num_samples)
        else:
            ns = types.SimpleNamespace(data_root=args.data_root, image_size=args.image_size)
            spec = "all" if args.init == "real" else ",".join(
                a[1:] if a.startswith("~") else f"~{a}" for a in feats)
            x0 = collect_real_images(get_filtered_dataset(ns, spec, split="test"),
                                     args.num_samples, seed=seed).to(device)

        singles = [apply(f, x0, args.batch_size) for f in nets]
        multi = run_cycles(nets, x0, args.max_iters, args.batch_size)
        pair = run_cycles(nets[:2], x0, args.max_iters, args.batch_size)

        r = {"seed": seed, "x0": measure(x0)[0],
             "single": {a: measure(h)[0] for a, h in zip(feats, singles)},
             "multi": [measure(x)[0] for x in multi],
             "pair": [measure(x)[0] for x in pair]}
        order_samples = {}
        if args.all_orders:
            # every order starts from the same x0, so each comparison is paired
            ref = [measure(x, residual=False)[1] for x in multi]
            r["orders"], r["mcnemar_p"] = {}, {}
            for perm in itertools.permutations(range(len(feats))):
                name = " -> ".join(feats[i] for i in perm)
                xs = run_cycles([nets[i] for i in perm], x0, args.max_iters, args.batch_size)
                outs = [measure(x, residual=False) for x in xs]
                r["orders"][name] = [m for m, _ in outs]
                r["mcnemar_p"][name] = [mcnemar(ref[k].cpu(), j.cpu())[2]
                                        for k, (_, j) in enumerate(outs)]
                order_samples[name] = xs[0][:16].cpu().clone()
                print(f"seed={seed} {name}: joint_all per cycle "
                      + " ".join(f"{m['joint_all']:.3f}" for m, _ in outs)
                      + f"  McNemar p vs given order (k=1) {r['mcnemar_p'][name][0]:.3g}",
                      flush=True)
        for k in range(args.max_iters):
            print(f"seed={seed} k={k + 1}  joint_all K-way={r['multi'][k]['joint_all']:.3f}  "
                  f"{feats[0]}&{feats[1]}: K-way={r['multi'][k][f'{feats[0]}&{feats[1]}']:.3f} "
                  f"2-way={r['pair'][k][f'{feats[0]}&{feats[1]}']:.3f}  "
                  f"residual={r['multi'][k]['joint_residual']:.4f}", flush=True)
        per_seed.append(r)

        if seed == args.seeds[0]:
            torch.save({"names": feats, "init": args.init, "x0": x0[:16].cpu().clone(),
                        "single": [h[:16].cpu().clone() for h in singles],
                        "multi": [x[:16].cpu().clone() for x in multi],
                        "pair": [x[:16].cpu().clone() for x in pair], "orders": order_samples},
                       os.path.join(args.out, "samples.pt"))

    results = {"config": vars(args), "device": str(device), "ckpts": {a: CKPTS[a] for a in feats},
               "seeds": per_seed, "summary": summarize(per_seed)}
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    samples = torch.load(os.path.join(args.out, "samples.pt"), weights_only=True)
    grid(samples, os.path.join(args.out, "grid.png"))
    rate_plot(results, os.path.join(args.out, "rates.png"))
    if args.all_orders:
        order_grid(samples, os.path.join(args.out, "orders_grid.png"))
        order_plot(results, os.path.join(args.out, "orders_rates.png"))
    print("done ->", args.out)


def tex(a):
    """Feature name for a mathtext subscript: ~Young -> \\neg Young, keeping underscores."""
    name = r"\mathrm{" + a.lstrip("~").replace("_", r"\_") + "}"
    return r"\neg " + name if a.startswith("~") else name


def summarize(per_seed):
    """mean / std over seeds of every scalar."""
    def ms(dicts):
        return {k: {"mean": float(np.mean([d[k] for d in dicts])),
                    "std": float(np.std([d[k] for d in dicts]))} for k in dicts[0]}
    out = {"x0": ms([r["x0"] for r in per_seed]),
           "single": {a: ms([r["single"][a] for r in per_seed]) for a in per_seed[0]["single"]},
           "multi": [ms([r["multi"][k] for r in per_seed]) for k in range(len(per_seed[0]["multi"]))],
           "pair": [ms([r["pair"][k] for r in per_seed]) for k in range(len(per_seed[0]["pair"]))]}
    if "orders" in per_seed[0]:
        out["orders"] = {o: [ms([r["orders"][o][k] for r in per_seed]) for k in range(len(its))]
                         for o, its in per_seed[0]["orders"].items()}
        # number of seeds on which this order differs significantly from the given one
        out["mcnemar_sig_seeds"] = {o: [sum(r["mcnemar_p"][o][k] < 0.05 for r in per_seed)
                                        for k in range(len(its))]
                                    for o, its in per_seed[0]["orders"].items()}
    return out


def grid(samples, path, n_cols=8, k_show=3):
    feats = samples["names"]
    chain = " → ".join(feats)
    rows = [("x$_0$ (start)", samples["x0"], None)]
    rows += [(f"$f_{{{tex(a)}}}$ only", h, None) for a, h in zip(feats, samples["single"])]
    rows.append((f"{feats[0]} → {feats[1]}\n1 cycle (2-way)", samples["pair"][0], COLORS[0]))
    for k in range(min(k_show, len(samples["multi"]))):
        rows.append((f"{chain}\n{k + 1} cycle{'s' if k else ''}", samples["multi"][k], COLORS[1]))

    image_rows(rows, path, n_cols,
               f"Alternating projection with {len(feats)} classes: {' + '.join(feats)}, "
               f"start = {samples['init']}\nblue frame = 2-way ({feats[0]} → {feats[1]}),  "
               f"red frame = {len(feats)}-way,  same x$_0$ in every column")


def order_grid(samples, path, n_cols=8):
    rows = [("x$_0$ (start)", samples["x0"], None)]
    rows += [(name.replace("->", "→") + "\n1 cycle", imgs, COLORS[i % len(COLORS)])
             for i, (name, imgs) in enumerate(samples["orders"].items())]
    image_rows(rows, path, n_cols,
               f"Every order of {' + '.join(samples['names'])}, 1 cycle, start = {samples['init']}"
               "\nsame x$_0$ in every column")


def image_rows(rows, path, n_cols, title):
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.25 * n_cols + 2.6, 1.3 * len(rows)))
    for r, (label, imgs, color) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(denorm_for_plot(imgs[c]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(color is not None)
                if color is not None:
                    s.set_color(color); s.set_linewidth(2.5)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    for c in range(n_cols):
        axes[0, c].set_title(f"sample {c + 1}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rate_plot(results, path):
    feats = results["config"]["features"]
    s = results["summary"]
    ks = list(range(1, len(s["multi"]) + 1))
    pair_key = f"{feats[0]}&{feats[1]}"
    panels = [("joint_all", f"joint ({' AND '.join(feats)})"),
              (pair_key, f"joint ({feats[0]} AND {feats[1]})")]
    panels += [(a, f"{a} rate") for a in feats]
    panels.append(("joint_residual", "joint fixed-point residual\nmax$_i$ |f$_i$(x) − x|"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8))
    for ax, (metric, title) in zip(axes, panels):
        series = [("multi", COLORS[1], f"{len(feats)}-way")]
        if metric in s["pair"][0] and metric != "joint_all":
            series.append(("pair", COLORS[0], f"2-way ({feats[0]} → {feats[1]})"))
        for key, color, lab in series:
            ax.errorbar(ks, [it[metric]["mean"] for it in s[key]],
                        yerr=[it[metric]["std"] for it in s[key]],
                        color=color, marker="o", capsize=3, label=lab)
        ax.axhline(s["x0"][metric]["mean"], color="#9ca3af", ls=":", label="x$_0$")
        for i, a in enumerate(feats):
            ax.axhline(s["single"][a][metric]["mean"], color=COLORS[2 + i % 3], ls="--", alpha=.6,
                       label=f"$f_{{{tex(a)}}}$ only")
        ax.set_title(title, fontsize=10); ax.set_xlabel("alternation cycles"); ax.set_xticks(ks)
        if metric != "joint_residual":
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("classifier rate")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{' + '.join(feats)}, start = {results['config']['init']}  "
                 f"(mean ± std over {len(results['seeds'])} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def order_plot(results, path):
    feats = results["config"]["features"]
    s = results["summary"]
    n_seeds = len(results["seeds"])
    ks = list(range(1, len(s["multi"]) + 1))
    panels = [("joint_all", f"joint ({' AND '.join(feats)})")] + [(a, f"{a} rate") for a in feats]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.9), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        for i, (name, its) in enumerate(s["orders"].items()):
            sig = s["mcnemar_sig_seeds"][name][0]
            ax.errorbar(ks, [it[metric]["mean"] for it in its], yerr=[it[metric]["std"] for it in its],
                        color=COLORS[i % len(COLORS)], marker="o", capsize=3, lw=2.5 if i == 0 else 1.5,
                        label=name.replace("->", "→") + ("  (given order)" if i == 0
                                      else f"  (sig. vs given at k=1: {sig}/{n_seeds} seeds)"))
        ax.axhline(s["x0"][metric]["mean"], color="#9ca3af", ls=":")
        ax.set_title(title, fontsize=10); ax.set_xlabel("alternation cycles"); ax.set_xticks(ks)
        ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    axes[0].set_ylabel("classifier rate")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"Every order, start = {results['config']['init']}  (mean ± std over {n_seeds} "
                 "seeds; dotted = x$_0$; sig. = McNemar p < 0.05 on the all-K joint outcome)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
