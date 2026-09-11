"""Cross-checkpoint experiments between two IGN runs, with visual outputs.

Given two runs (e.g. Male and Not_Young), this script loads several
checkpoints of each and renders:

  1. matrix_<A>_then_<B>_sample<k>.png (and the reverse order)
       The composition sweep: cell (i, j) = second model's epoch-j ckpt
       applied to first model's epoch-i output, for one input image x.
       Rows = first model's epochs, columns = second's (listed in
       results.json and the console). One file per sample image.

  2. sharpen_<run>_e<early>_then_e<late>.png
       Within one run: rows x, f_early(x), f_late(f_early(x)), f_late(x) --
       does the late (sharp autoencoder) checkpoint clean up the early
       (strong projector) checkpoint's blurry output without undoing it?

  3. alt_A<ea>_B<eb>.png / alt_B<eb>_A<ea>.png [/ alt_..._from_noise.png]
       Trajectory of the plain alternating iteration with one chosen pair:
       rows x, cycle 1, cycle 2, ... The noise variant starts from
       FFT-matched noise when run A's checkpoint carries freq_stats.

  3b. alt_sched_<tokens>.png
       Interleaved projector/sharpener schedules mixing BOTH runs' early
       (e = strong projector) and late (l = sharp autoencoder) checkpoints,
       e.g. alt_sched_Ae50-Bl475-Be50-Al475.png applies A-early, B-late,
       B-early, A-late per cycle. Row n = after n full cycles.

  4. heatmap_success.png + results.json (needs the attribute classifier)
       For every checkpoint pair and both orders: fraction of a lacks-both
       test batch that the classifier scores as having BOTH target features
       after the composition. The quantitative version of (1).

Inputs are TEST-split images lacking both features (ground-truth labels),
forwarded with train-mode BatchNorm as full batches. Nothing is trained.

    python cross_checkpoint_experiments.py \
        --run_a runs/Male_paper500 --run_b runs/Not_Young_500
    python cross_checkpoint_experiments.py --run_a ... --run_b ... \
        --epochs_a 25,50,100,475 --epoch_a 50 --epoch_b 50
"""

import argparse
import glob
import json
import os
import types

import torch

from ign_data import CELEBA_ATTRS, get_filtered_dataset, parse_attr_spec
from ign_models import load_classifier, load_ign
from ign_utils import sample_freq_noise, save_image_grid, set_seed


# ---------------------------------------------------------------- helpers

def find_ckpts(run_dir):
    by_epoch = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt"))):
        e = torch.load(p, map_location="cpu", weights_only=False)["epoch"]
        by_epoch.setdefault(e, p)
    if not by_epoch:
        raise SystemExit(f"no ckpt_*.pt in {run_dir}")
    return by_epoch


def pick_epochs(available, spec, cap=5):
    """'auto' -> up to `cap` epochs spread over the run (skipping epoch 0)."""
    if spec != "auto":
        want = [int(s) for s in spec.split(",")]
        missing = [e for e in want if e not in available]
        if missing:
            raise SystemExit(f"epochs {missing} not found; available: {available}")
        return want
    usable = [e for e in available if e > 0] or available
    if len(usable) <= cap:
        return usable
    idx = [round(i * (len(usable) - 1) / (cap - 1)) for i in range(cap)]
    return sorted({usable[i] for i in idx})


def load_models(by_epoch, epochs, device):
    models = {}
    for e in epochs:
        m, _ = load_ign(by_epoch[e], device)
        m.train()  # batch-statistics BN; inference only, models are discarded at exit
        models[e] = m
    return models


def feature_of(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    idx, want = parse_attr_spec(ckpt.get("attr_spec", "Smiling"))[0]
    return idx, want, ckpt


@torch.no_grad()
def apply_seq(models_seq, x):
    for m in models_seq:
        x = m(x)
    return x


# ------------------------------------------------------------- experiments

@torch.no_grad()
def composition_matrix(models_a, models_b, x, out_dir, tag, n_samples):
    """cell (i, j) = models_b[ej](models_a[ei](x)); one PNG per sample image."""
    ea, eb = sorted(models_a), sorted(models_b)
    a_out = {i: models_a[i](x) for i in ea}
    cells = {(i, j): models_b[j](a_out[i]) for i in ea for j in eb}
    for k in range(n_samples):
        rows = [cells[(i, j)][k] for i in ea for j in eb]
        save_image_grid(torch.stack(rows), os.path.join(out_dir, f"matrix_{tag}_sample{k}.png"),
                        nrow=len(eb))
    return {f"{i}->{j}": None for i in ea for j in eb}  # keys documented in results


@torch.no_grad()
def sharpen_grid(models, x, out_path, show):
    """rows: x, f_early(x), f_late(f_early(x)), f_late(x)."""
    es = sorted(models)
    early, late = models[es[0]], models[es[-1]]
    fe = early(x)
    rows = [x[:show], fe[:show], late(fe)[:show], late(x)[:show]]
    save_image_grid(torch.cat(rows, 0), out_path, nrow=show)
    return es[0], es[-1]


@torch.no_grad()
def schedule_grid(seq, x, iters, out_path, show):
    """One full cycle = applying every model in `seq` in order. Grid rows:
    x, then the state after each completed cycle."""
    rows = [x[:show]]
    cur = x
    for _ in range(iters):
        for m in seq:
            cur = m(cur)
        rows.append(cur[:show])
    save_image_grid(torch.cat(rows, 0), out_path, nrow=show)


@torch.no_grad()
def success_heatmaps(models_a, models_b, x, clf, feats, names, out_dir):
    """fraction of the batch with BOTH features present after the composition."""
    ea, eb = sorted(models_a), sorted(models_b)
    (ia, wa), (ib, wb) = feats
    name_a, name_b = names
    tags = {f"{name_a}_then_{name_b}": (models_a, models_b, ea, eb),
            f"{name_b}_then_{name_a}": (models_b, models_a, eb, ea)}

    def joint_rate(imgs):
        p = torch.sigmoid(clf(imgs))
        pa = p[:, ia] if wa == 1 else 1 - p[:, ia]
        pb = p[:, ib] if wb == 1 else 1 - p[:, ib]
        return float(((pa > 0.5) & (pb > 0.5)).float().mean())

    mats = {}
    for tag, (first, second, e1, e2) in tags.items():
        mats[tag] = [[joint_rate(second[j](first[i](x))) for j in e2] for i in e1]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, tag in zip(axes, tags):
            _, _, r_lab, c_lab = tags[tag]
            im = ax.imshow(mats[tag], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"joint rate: {tag}", fontsize=9)
            ax.set_xticks(range(len(c_lab)), c_lab)
            ax.set_yticks(range(len(r_lab)), r_lab)
            ax.set_xlabel("second model epoch")
            ax.set_ylabel("first model epoch")
            for i in range(len(r_lab)):
                for j in range(len(c_lab)):
                    ax.text(j, i, f"{mats[tag][i][j]:.2f}", ha="center", va="center",
                            color="w", fontsize=8)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "heatmap_success.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"(heatmap plot skipped: {e})")
    return mats


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser("cross-checkpoint experiments between two IGN runs")
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--epochs_a", default="auto", help="'auto' or comma list, e.g. 25,50,100")
    ap.add_argument("--epochs_b", default="auto")
    ap.add_argument("--max_epochs", type=int, default=5, help="cap for 'auto' selection")
    ap.add_argument("--epoch_a", type=int, default=None,
                    help="A-epoch for the alternating trajectory (default: earliest picked)")
    ap.add_argument("--epoch_b", type=int, default=None)
    ap.add_argument("--iters", type=int, default=6, help="alternating cycles shown")
    ap.add_argument("--classifier", default="runs/classifier/classifier.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--show", type=int, default=8, help="columns in strip grids")
    ap.add_argument("--matrix_samples", type=int, default=3,
                    help="how many single-image matrix PNGs")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_a, name_b = (os.path.basename(os.path.normpath(p)) for p in (args.run_a, args.run_b))
    out_dir = args.out_dir or os.path.join("cross_experiments", f"{name_a}__{name_b}")
    os.makedirs(out_dir, exist_ok=True)

    ck_a, ck_b = find_ckpts(args.run_a), find_ckpts(args.run_b)
    ea = pick_epochs(sorted(ck_a), args.epochs_a, args.max_epochs)
    eb = pick_epochs(sorted(ck_b), args.epochs_b, args.max_epochs)
    ia, wa, first_a = feature_of(ck_a[ea[0]])
    ib, wb, _ = feature_of(ck_b[eb[0]])
    print(f"A = {name_a}: feature {CELEBA_ATTRS[ia]}={wa}, epochs {ea}")
    print(f"B = {name_b}: feature {CELEBA_ATTRS[ib]}={wb}, epochs {eb}")

    # test images lacking BOTH features (w.r.t. the trained direction)
    ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
    ds = get_filtered_dataset(ns, "all", split="test")
    attr = ds.base.attr
    mask = (attr[:, ia] != wa) & (attr[:, ib] != wb)
    pool = mask.nonzero(as_tuple=True)[0]
    if len(pool) < args.batch:
        print(f"WARNING: only {len(pool)} lacks-both test images; padding with random")
        pool = torch.cat([pool, torch.randperm(len(ds))[: args.batch - len(pool)]])
    g = torch.Generator().manual_seed(args.seed)
    sel = pool[torch.randperm(len(pool), generator=g)[:args.batch]]
    x = torch.stack([ds[i][0] for i in sel.tolist()]).to(device)
    print(f"input: {len(x)} test images lacking both features")

    models_a = load_models(ck_a, ea, device)
    models_b = load_models(ck_b, eb, device)

    # 1) composition matrices, both orders (apply first model, then second)
    composition_matrix(models_a, models_b, x, out_dir,
                       f"{name_a}_then_{name_b}", args.matrix_samples)
    composition_matrix(models_b, models_a, x, out_dir,
                       f"{name_b}_then_{name_a}", args.matrix_samples)
    print("matrix grids: rows = first model's epochs, cols = second's (as printed)")

    # 2) early-projector + late-sharpener, per run
    se = sharpen_grid(models_a, x,
                      os.path.join(out_dir, f"sharpen_{name_a}_e{ea[0]}_then_e{ea[-1]}.png"),
                      args.show)
    se = sharpen_grid(models_b, x,
                      os.path.join(out_dir, f"sharpen_{name_b}_e{eb[0]}_then_e{eb[-1]}.png"),
                      args.show)

    # 3) alternating trajectories with one chosen pair
    pa = args.epoch_a if args.epoch_a is not None else ea[0]
    pb = args.epoch_b if args.epoch_b is not None else eb[0]
    fa = models_a[pa] if pa in models_a else load_models(ck_a, [pa], device)[pa]
    fb = models_b[pb] if pb in models_b else load_models(ck_b, [pb], device)[pb]
    schedule_grid([fa, fb], x, args.iters,
                  os.path.join(out_dir, f"alt_A{pa}_B{pb}.png"), args.show)
    schedule_grid([fb, fa], x, args.iters,
                  os.path.join(out_dir, f"alt_B{pb}_A{pa}.png"), args.show)
    print(f"alternating grids use A e{pa}, B e{pb}")
    if "freq_stats" in first_a:
        stats = {k: v.to(device) for k, v in first_a["freq_stats"].items()}
        z = sample_freq_noise(stats, args.batch)
        schedule_grid([fa, fb], z, args.iters,
                      os.path.join(out_dir, f"alt_A{pa}_B{pb}_from_noise.png"), args.show)
        print("alt_*_from_noise.png: started from FFT-matched noise")

    # 3b) interleaved projector/sharpener schedules across both runs.
    # Ae/Al = A's earliest/latest picked epoch (strong projector / sharp
    # autoencoder), same for B. One grid per schedule; row n = after n cycles.
    tok = {"Ae": (models_a[ea[0]], f"Ae{ea[0]}"), "Al": (models_a[ea[-1]], f"Al{ea[-1]}"),
           "Be": (models_b[eb[0]], f"Be{eb[0]}"), "Bl": (models_b[eb[-1]], f"Bl{eb[-1]}")}
    schedules = [
        ["Ae", "Be"],                    # both projectors (baseline)
        ["Al", "Bl"],                    # both sharpeners (weak-transfer baseline)
        ["Ae", "Bl", "Be", "Al"],        # project A, sharpen w/ B-late, project B, sharpen w/ A-late
        ["Ae", "Al", "Be", "Bl"],        # project-then-sharpen per feature
        ["Ae", "Be", "Al", "Bl"],        # project both, then sharpen both
    ]
    sched_names = []
    for s in schedules:
        label = "-".join(tok[t][1] for t in s)
        schedule_grid([tok[t][0] for t in s], x, args.iters,
                      os.path.join(out_dir, f"alt_sched_{label}.png"), args.show)
        sched_names.append(label)
    print("schedule grids: " + ", ".join(sched_names))

    # 4) quantitative sweep, if the classifier is available
    results = {"A": {"run": args.run_a, "feature": f"{CELEBA_ATTRS[ia]}={wa}", "epochs": ea},
               "B": {"run": args.run_b, "feature": f"{CELEBA_ATTRS[ib]}={wb}", "epochs": eb},
               "alternating_pair": [pa, pb]}
    results["schedules"] = sched_names
    if os.path.exists(args.classifier):
        clf, _ = load_classifier(args.classifier, device)
        mats = success_heatmaps(models_a, models_b, x, clf, ((ia, wa), (ib, wb)),
                                (name_a, name_b), out_dir)
        results["joint_rates"] = mats
        ab = mats[f"{name_a}_then_{name_b}"]
        best = max(((v, i, j) for i, row in zip(ea, ab) for j, v in zip(eb, row)))
        print(f"best {name_a}->{name_b} pair by joint rate: "
              f"A e{best[1]} -> B e{best[2]} ({best[0]:.2f})")
    else:
        print(f"no classifier at {args.classifier}: heatmaps skipped")

    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nall outputs -> {out_dir}")


if __name__ == "__main__":
    main()
