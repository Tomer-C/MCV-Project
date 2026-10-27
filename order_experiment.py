"""Does the order of application matter in alternating projection?

For orthogonal projections onto subspaces, von Neumann's iteration reaches the
same limit whichever projector goes first. IGNs are non-linear and only
approximately idempotent, so that guarantee is gone; this script measures what
actually happens.

For each feature pair (A, B) and each starting distribution x_0, the SAME x_0
batch is pushed through both orders,

    A->B :  x_{n+1} = f_B(f_A(x_n))
    B->A :  x_{n+1} = f_A(f_B(x_n))

for k = 1..K full cycles, and at every k we record

  * joint and per-attribute classifier rates for each order
    (per-attribute rates test the "last network applied wins" hypothesis),
  * a paired McNemar test on the joint outcome (same x_0, so the comparison
    is paired, not two independent samples),
  * the image distance between the two orders (L2 / PSNR / SSIM),
  * a NOISE FLOOR for that distance: the same order run twice with the batch
    composition shuffled. BatchNorm runs on batch statistics (the only mode
    upstream IGN uses), so outputs depend slightly on which images share a
    batch; order differences must be judged against this floor,
  * distance of each order's output to the trained intersection baseline,
    when one exists,
  * FID vs real test-set intersection images (CelebA needed).

Several seeds are run; tables report mean +- std over seeds. Labeled image
grids are drawn from seed 0.

Example
-------
  python order_experiment.py --runs_dir runs --data_root ./data \
      --num_samples 1024 --seeds 0 1 2 --max_iters 4
"""

import argparse
import json
import math
import os
import types

import numpy as np
import torch

from alternating import set_inference_mode
from ign_data import ATTR_TO_IDX, _celeba_files_present, get_filtered_dataset
from ign_models import load_classifier, load_ign
from ign_utils import freq_stats_from_images, sample_freq_noise, set_seed
from metrics import (FeatureExtractor, attr_probabilities, collect_real_images,
                     frechet_distance, paired_distances)

# Projector checkpoints used for the team's headline results (RESULTS.md):
# early-epoch checkpoints, where projection strength peaks.
CKPTS = {"Male": "Male_500_withsamples/ckpt_0075.pt",
         "Smiling": "Smiling_paper500/ckpt_0025.pt",
         "Blond_Hair": "Blond_Hair_500/ckpt_0075.pt"}
BASELINES = {("Male", "Smiling"): "Baseline_Male_AND_Smiling/ckpt_last.pt"}
PAIRS = [("Male", "Smiling"), ("Male", "Blond_Hair")]
# out-of-distribution CelebA photos: attribute spec templates per pair (A, B)
REAL_INITS = {"neither": "~{A},~{B}", "only_A": "{A},~{B}", "only_B": "~{A},{B}"}


def get_args_parser():
    p = argparse.ArgumentParser("Order-of-application experiment")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--classifier", default=None,
                   help="defaults to <runs_dir>/classifier/classifier.pt")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="pairs as A,B (e.g. Male,Smiling); default: Male+Smiling, Male+Blond_Hair")
    p.add_argument("--inits", nargs="+", default=["noise", "neither"],
                   choices=["noise", "real"] + list(REAL_INITS),
                   help="noise = FFT-matched noise; real = any CelebA test photo; "
                        "neither / only_A / only_B = test photos outside the target, "
                        "lacking both features / having only A / having only B")
    p.add_argument("--ckpt", nargs="+", default=[],
                   help="override projector checkpoints, e.g. Male=Male_500_withsamples/ckpt_0300.pt")
    p.add_argument("--noise_stats", default="ckpt",
                   help="FFT stats for noise x_0: 'ckpt' = stored in the first net's checkpoint, "
                        "'data' = computed from the pair's intersection train images (needs "
                        "CelebA train), or a checkpoint path under runs_dir to take them from "
                        "(keeps x_0 identical across checkpoint sets)")
    p.add_argument("--num_samples", default=1024, type=int)
    p.add_argument("--seeds", nargs="+", default=[0, 1, 2], type=int)
    p.add_argument("--max_iters", default=4, type=int)
    p.add_argument("--batch_size", default=256, type=int,
                   help="BatchNorm batch size; identical for both orders")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--out", default=os.path.join("exp_results", "order_experiment"))
    return p


@torch.no_grad()
def apply(model, x, bs, perm=None):
    """Forward in fixed chunks of `bs`. With `perm`, chunk membership is
    shuffled (and the output un-shuffled), which changes only the BatchNorm
    batch statistics -- used for the noise-floor control."""
    if perm is not None:
        x = x[perm]
    y = torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])
    if perm is not None:
        y = y[torch.argsort(perm)]
    return y


@torch.no_grad()
def run_order(first, second, x0, k_max, bs, perm=None):
    """Iterates after every full cycle, plus the half-step f_first(x0)."""
    xs, x = [], x0
    half = None
    for k in range(k_max):
        h = apply(first, x, bs, perm)
        if k == 0:
            half = h
        x = apply(second, h, bs, perm)
        xs.append(x)
    return half, xs


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans. Returns (n01, n10, p)."""
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    # exact binomial tail, two-sided, in log space to stay stable for large n
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return n01, n10, min(1.0, 2 * tail)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    have_data = _celeba_files_present(args.data_root)
    if not have_data:
        print(f"[order] CelebA not found under {args.data_root}/celeba -- real-image init "
              "and FID are skipped; noise uses checkpoint FFT stats.")
    inits = [i for i in args.inits if have_data or i == "noise"]
    ns = types.SimpleNamespace(data_root=args.data_root, image_size=args.image_size)

    clf, _ = load_classifier(args.classifier or os.path.join(args.runs_dir, "classifier",
                                                              "classifier.pt"), device)
    pairs = [tuple(s.split(",")) for s in args.pairs] if args.pairs else PAIRS
    CKPTS.update(dict(s.split("=", 1) for s in args.ckpt))
    nets, ckpts = {}, {}
    for name in {n for pr in pairs for n in pr}:
        nets[name], ckpts[name] = load_ign(os.path.join(args.runs_dir, CKPTS[name]), device)
    baselines = {}
    for pr in pairs:
        if pr in BASELINES and os.path.exists(os.path.join(args.runs_dir, BASELINES[pr])):
            baselines[pr], _ = load_ign(os.path.join(args.runs_dir, BASELINES[pr]), device)
    set_inference_mode(list(nets.values()) + list(baselines.values()), eval_mode=False)

    extractor = None
    if have_data and not args.skip_fid:
        extractor = FeatureExtractor(device, fallback_classifier=clf)

    def rates(imgs, ia, ib):
        p = attr_probabilities(clf, imgs, device)
        a, b = p[:, ia] > 0.5, p[:, ib] > 0.5
        return a, b, a & b

    results = {"config": vars(args), "device": str(device), "have_celeba": have_data,
               "ckpts": CKPTS, "baselines": {f"{a}+{b}": v for (a, b), v in BASELINES.items()}}
    for A, B in pairs:
        ia, ib = ATTR_TO_IDX[A], ATTR_TO_IDX[B]
        spec = f"{A},{B}"
        f_real = None
        if extractor is not None:
            ds_test = get_filtered_dataset(ns, spec, split="test")
            f_real = extractor(collect_real_images(ds_test, min(args.num_samples, len(ds_test)), seed=0))

        for init in inits:
            tag = f"{A}+{B}__{init}"
            per_seed = []
            for seed in args.seeds:
                set_seed(seed)
                if init == "noise":
                    if have_data and args.noise_stats == "data":
                        # FFT stats of the target intersection, as in the team's recipes
                        ds = get_filtered_dataset(ns, spec, split="train")
                        g = torch.Generator().manual_seed(seed)
                        sel = torch.randperm(len(ds), generator=g)[:min(256, len(ds))]
                        imgs = torch.stack([ds[i][0] for i in sel.tolist()]).to(device)
                        stats = freq_stats_from_images(imgs)
                    elif args.noise_stats not in ("ckpt", "data"):
                        src = torch.load(os.path.join(args.runs_dir, args.noise_stats),
                                         map_location=device, weights_only=False)
                        stats = {k: v.to(device) for k, v in src["freq_stats"].items()}
                    else:
                        stats = {k: v.to(device) for k, v in ckpts[A]["freq_stats"].items()}
                    x0 = sample_freq_noise(stats, args.num_samples)
                else:
                    init_spec = "all" if init == "real" else REAL_INITS[init].format(A=A, B=B)
                    ds = get_filtered_dataset(ns, init_spec, split="test")
                    x0 = collect_real_images(ds, args.num_samples, seed=seed).to(device)

                fA, fB = nets[A], nets[B]
                half_ab, ab = run_order(fA, fB, x0, args.max_iters, args.batch_size)
                half_ba, ba = run_order(fB, fA, x0, args.max_iters, args.batch_size)
                g = torch.Generator().manual_seed(1000 + seed)
                perm = torch.randperm(len(x0), generator=g).to(device)
                _, ab_ctrl = run_order(fA, fB, x0, args.max_iters, args.batch_size, perm)
                base = apply(baselines[(A, B)], x0, args.batch_size) if (A, B) in baselines else None

                r = {"seed": seed, "x0": {}, "single": {}, "iters": []}
                a0, b0, j0 = rates(x0, ia, ib)
                r["x0"] = {A: a0.float().mean().item(), B: b0.float().mean().item(),
                           "joint": j0.float().mean().item()}
                for nm, h in ((A, half_ab), (B, half_ba)):
                    a, b, j = rates(h, ia, ib)
                    r["single"][nm] = {A: a.float().mean().item(), B: b.float().mean().item(),
                                       "joint": j.float().mean().item()}
                if base is not None:
                    a, b, j = rates(base, ia, ib)
                    r["baseline"] = {A: a.float().mean().item(), B: b.float().mean().item(),
                                     "joint": j.float().mean().item()}
                    if f_real is not None:
                        r["baseline"]["fid"] = frechet_distance(f_real, extractor(base.cpu()), warn_rank=False)

                for k in range(args.max_iters):
                    it = {"k": k + 1}
                    outs = {}
                    for order, x in (("A->B", ab[k]), ("B->A", ba[k])):
                        a, b, j = rates(x, ia, ib)
                        outs[order] = j
                        it[order] = {A: a.float().mean().item(), B: b.float().mean().item(),
                                     "joint": j.float().mean().item()}
                        if f_real is not None:
                            it[order]["fid"] = frechet_distance(f_real, extractor(x.cpu()), warn_rank=False)
                        if base is not None:
                            d = paired_distances(x.cpu(), base.cpu(), use_lpips=False)
                            it[order]["vs_baseline"] = {kk: d[kk] for kk in ("l2_mean", "psnr_mean", "ssim_mean")}
                    n01, n10, p = mcnemar(outs["A->B"].cpu(), outs["B->A"].cpu())
                    it["mcnemar"] = {"only_B->A_joint": n01, "only_A->B_joint": n10, "p": p}
                    it["agree_joint"] = (outs["A->B"] == outs["B->A"]).float().mean().item()
                    d = paired_distances(ab[k].cpu(), ba[k].cpu(), use_lpips=False)
                    it["order_distance"] = {kk: d[kk] for kk in ("l2_mean", "psnr_mean", "ssim_mean")}
                    d = paired_distances(ab[k].cpu(), ab_ctrl[k].cpu(), use_lpips=False)
                    it["bn_noise_floor"] = {kk: d[kk] for kk in ("l2_mean", "psnr_mean", "ssim_mean")}
                    # which single net's output does each order end up resembling?
                    for order, x in (("A->B", ab[k]), ("B->A", ba[k])):
                        for nm, h in ((A, half_ab), (B, half_ba)):
                            d = paired_distances(x.cpu(), h.cpu(), use_lpips=False)
                            it[order][f"ssim_to_f{nm}(x0)"] = d["ssim_mean"]
                            it[order][f"l2_to_f{nm}(x0)"] = d["l2_mean"]
                    r["iters"].append(it)
                    print(f"{tag} seed={seed} k={k + 1}  "
                          f"joint A->B={it['A->B']['joint']:.3f} B->A={it['B->A']['joint']:.3f} "
                          f"(p={p:.3g})  L2 order={it['order_distance']['l2_mean']:.2f} "
                          f"floor={it['bn_noise_floor']['l2_mean']:.2f}", flush=True)
                per_seed.append(r)

                if seed == args.seeds[0]:
                    torch.save({"x0": x0[:16].cpu(), "half_ab": half_ab[:16].cpu(),
                                "half_ba": half_ba[:16].cpu(),
                                "ab": [x[:16].cpu() for x in ab], "ba": [x[:16].cpu() for x in ba],
                                "baseline": None if base is None else base[:16].cpu(),
                                "names": (A, B), "init": init,
                                "init_desc": init_description(init, A, B)},
                               os.path.join(args.out, f"{tag}_samples.pt"))
            results[tag] = {"pair": [A, B], "init": init,
                            "init_desc": init_description(init, A, B), "seeds": per_seed,
                            "summary": summarize(per_seed, A, B)}

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", args.out)


def init_description(init, A, B):
    return {"noise": "FFT-matched noise",
            "real": "CelebA test photos (any)",
            "neither": f"CelebA test photos with neither {A} nor {B}",
            "only_A": f"CelebA test photos: {A}, not {B}",
            "only_B": f"CelebA test photos: {B}, not {A}"}[init]


def summarize(per_seed, A, B):
    """mean / std over seeds of every scalar, per iteration."""
    def ms(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    out = {"x0": {k: ms([r["x0"][k] for r in per_seed]) for k in (A, B, "joint")},
           "single": {nm: {k: ms([r["single"][nm][k] for r in per_seed]) for k in (A, B, "joint")}
                      for nm in (A, B)},
           "iters": []}
    if "baseline" in per_seed[0]:
        out["baseline"] = {k: ms([r["baseline"][k] for r in per_seed]) for k in per_seed[0]["baseline"]}
    for k in range(len(per_seed[0]["iters"])):
        its = [r["iters"][k] for r in per_seed]
        s = {"k": k + 1}
        for order in ("A->B", "B->A"):
            s[order] = {m: ms([it[order][m] for it in its])
                        for m in its[0][order] if not isinstance(its[0][order][m], dict)}
            if "vs_baseline" in its[0][order]:
                s[order]["vs_baseline"] = {m: ms([it[order]["vs_baseline"][m] for it in its])
                                           for m in its[0][order]["vs_baseline"]}
        s["joint_diff_A->B_minus_B->A"] = ms([it["A->B"]["joint"] - it["B->A"]["joint"] for it in its])
        s["mcnemar_p"] = [it["mcnemar"]["p"] for it in its]
        s["agree_joint"] = ms([it["agree_joint"] for it in its])
        for key in ("order_distance", "bn_noise_floor"):
            s[key] = {m: ms([it[key][m] for it in its]) for m in its[0][key]}
        out["iters"].append(s)
    return out


if __name__ == "__main__":
    main(get_args_parser().parse_args())
