"""Evaluate a commuting-loss sweep against plain (naive) alternating projection.

For every run directory produced by train/train_commuting_pair.py -- and every
ckpt_{A,B}_eNNNN.pt snapshot inside it -- plus the untouched warm-start pair
("naive", epoch 0), this runs x_{n+1} = f_B(f_A(x_n)) for n = 1..max_iters on
one fixed x_0 per domain and records:

  joint / per-attribute rate  classifier-judged; the headline metric
  order distance              ||(f_B f_A)^n x0 - (f_A f_B)^n x0||, the premise
                              of the commuting-idempotents theorem
  single-net rates            f_A(x0), f_B(x0) alone: does each net still project?
  sharpness                   mean |Laplacian| relative to real faces -- the first
                              commuting probe collapsed toward blurry mean faces
  diversity                   per-pixel std across the batch over the central face
                              crop, relative to real faces; ~0 when every x_0 lands
                              on the same face, blurry or not (the LPIPS variant
                              collapses to a *sharp* one). The crop keeps varied
                              backgrounds/hair from masking a collapsed face.
  joint residual              max_i ||f_i(x) - x||, distance from a joint fixed point
  KID / FID (--kid)           Inception-feature distance to real A-and-B test photos.
                              The classifier happily accepts one blurry prototype
                              "A-and-B" face for every input, so joint rate alone
                              can reward collapse; KID cannot be gamed that way

Two domains, as in docs/method3_commuting.md: 'real' (real test faces,
in-distribution) and 'fftnoise' (FFT-matched noise built from train images of
--noise_stats_spec, out-of-distribution). BatchNorm runs in train mode with
frozen running stats over the whole x_0 batch, like experiments/method1_single_ign/run_experiment.py.

x_0 is built once and cached in <out>/x0.pt, and results accumulate in
<out>/results.json (entries already there are skipped unless --force), so the
script can be re-run as sweep runs finish and every entry shares the same x_0.
experiments/method3_commuting/make_commuting_grids.py draws the image grids from <out>/grid_tensors.pt.

Example
-------
  python -m experiments.method3_commuting.commuting_sweep_eval \\
      --naive_a runs/Male_500_withsamples/ckpt_0075.pt \\
      --naive_b runs/Smiling_paper500/ckpt_0075.pt \\
      --baseline runs/Baseline_Male_AND_Smiling/ckpt_last.pt \\
      --intersection_spec "Male,Smiling" \\
      --runs runs/sweep_ms_* --out exp_results/commuting_sweep/male_smiling
"""

import argparse
import csv
import glob
import json
import os
import re

import torch
import torch.nn.functional as F

from ign.alternating import set_inference_mode
from ign.data import get_filtered_dataset
from ign.models import load_classifier, load_ign
from ign.utils import freq_stats_from_images, resolve_device, sample_freq_noise, set_seed
from ign.metrics import (FeatureExtractor, attr_rates, collect_real_images,
                     frechet_distance, kernel_distance)

DOMAINS = ("real", "fftnoise")


def get_args_parser():
    p = argparse.ArgumentParser("Evaluate a commuting-loss sweep")
    p.add_argument("--naive_a", required=True, help="warm-start checkpoint for f_A")
    p.add_argument("--naive_b", required=True, help="warm-start checkpoint for f_B")
    p.add_argument("--baseline", default=None, help="f_AnB trained on the intersection")
    p.add_argument("--intersection_spec", required=True)
    p.add_argument("--classifier", default="runs/classifier/classifier.pt")
    p.add_argument("--runs", nargs="*", default=[],
                   help="train/train_commuting_pair.py run dirs (globs allowed)")
    p.add_argument("--epochs", nargs="*", type=int, default=None,
                   help="only these snapshot epochs (default: all found)")
    p.add_argument("--out", required=True)
    p.add_argument("--num_samples", default=256, type=int)
    p.add_argument("--max_iters", default=3, type=int)
    p.add_argument("--grid_n", default=8, type=int, help="images per entry kept for grids")
    p.add_argument("--noise_stats_spec", default="all")
    p.add_argument("--force", action="store_true", help="re-evaluate cached entries")
    p.add_argument("--kid", action="store_true",
                   help="also KID/FID of every iterate vs real intersection test images")
    p.add_argument("--kid_ref_n", default=2048, type=int,
                   help="real intersection images used as the KID/FID reference")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", default=0, type=int)
    return p


def build_x0(args, device):
    """Same construction as experiments/method1_single_ign/run_experiment.py's --init real and
    --init noise --noise_stats_spec, with its own fixed seeds."""
    ds = get_filtered_dataset(args, "all", split="test")
    real = collect_real_images(ds, args.num_samples, seed=args.seed)

    ds_stats = get_filtered_dataset(args, args.noise_stats_spec, split="train")
    g = torch.Generator().manual_seed(args.seed)
    sel = torch.randperm(len(ds_stats), generator=g)[:256]
    imgs = torch.stack([ds_stats[i][0] for i in sel.tolist()])
    torch.manual_seed(args.seed)
    noise = sample_freq_noise(freq_stats_from_images(imgs), args.num_samples)
    return {"real": real.to(device), "fftnoise": noise.to(device)}


_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


@torch.no_grad()
def sharpness(x):
    """Mean |Laplacian| of the grayscale image in [0, 1]: high-frequency detail."""
    gray = ((x.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
    return F.conv2d(gray, _LAPLACE.to(x.device)).abs().mean().item()


@torch.no_grad()
def diversity(x):
    """Mean per-pixel std across the batch over the central half of the image
    (the aligned face): ~0 when all outputs share one face."""
    h, w = x.shape[-2:]
    face = x[..., h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return face.clamp(-1, 1).std(0).mean().item()


@torch.no_grad()
def l2_per_image(a, b):
    return (a - b).flatten(1).norm(dim=1).mean().item()


@torch.no_grad()
def joint_residual(models, x):
    return torch.stack([(f(x) - x).abs().flatten(1).mean(1) for f in models], 0) \
        .max(0).values.mean().item()


@torch.no_grad()
def evaluate_pair(f_a, f_b, x0s, classifier, spec, device, max_iters, grid_n, refs,
                  dist_fn=None):
    """Metrics + grid tensors for one (f_A, f_B) pair on every domain."""
    set_inference_mode([f_a, f_b], eval_mode=False)
    out, grids = {}, {}
    for dom, x0 in x0s.items():
        ab, ba = x0.clone(), x0.clone()   # ab = (f_B o f_A)^n x0, applies A first
        iters, grid = [], []
        for n in range(1, max_iters + 1):
            ab = f_b(f_a(ab))
            ba = f_a(f_b(ba))
            r = attr_rates(classifier, ab.cpu(), spec, device)
            iters.append({
                "n": n, "joint": r["joint"], "per_attr": r["per_attr"],
                "mean_prob": r["mean_prob"],
                "joint_reverse_order": attr_rates(classifier, ba.cpu(), spec, device)["joint"],
                "order_dist": l2_per_image(ab, ba),
                "sharpness": sharpness(ab),
                "sharpness_rel": sharpness(ab) / refs["sharpness"],
                "diversity_rel": diversity(ab) / refs["diversity"],
                "joint_residual": joint_residual([f_a, f_b], ab),
                **(dist_fn(ab) if dist_fn else {}),
            })
            grid.append(ab[:grid_n].half().cpu())
        single_a = attr_rates(classifier, f_a(x0).cpu(), spec, device)
        single_b = attr_rates(classifier, f_b(x0).cpu(), spec, device)
        out[dom] = {"iters": iters, "single_A": single_a, "single_B": single_b}
        grids[dom] = grid
    return out, grids


def find_snapshots(run_dir):
    """{epochs_done: (ckpt_A, ckpt_B)} for every snapshot pair in a run dir.
    Runs from before per-epoch snapshots existed only have ckpt_{A,B}_last.pt,
    written at the final epoch; those count as epoch num_epochs."""
    last = [os.path.join(run_dir, f"ckpt_{n}_last.pt") for n in "AB"]
    if not glob.glob(os.path.join(run_dir, "ckpt_A_e*.pt")) and all(map(os.path.exists, last)):
        return {run_config(run_dir).get("num_epochs") or 0: tuple(last)}
    snaps = {}
    for pa in glob.glob(os.path.join(run_dir, "ckpt_A_e*.pt")):
        m = re.search(r"ckpt_A_e(\d+)\.pt$", pa)
        if not m:
            continue
        pb = os.path.join(run_dir, f"ckpt_B_e{m.group(1)}.pt")
        if os.path.exists(pb):
            snaps[int(m.group(1))] = (pa, pb)
    return dict(sorted(snaps.items()))


def run_config(run_dir):
    try:
        with open(os.path.join(run_dir, "args.json")) as fh:
            a = json.load(fh)
    except FileNotFoundError:
        return {}
    keys = ("lambda_commute", "lr", "commute_metric", "num_epochs", "attr_a", "attr_b",
            "ckpt_a", "ckpt_b", "discriminative_loss_factor", "seed")
    return {k: a.get(k) for k in keys}


def write_csv(results, path):
    rows = []
    for key, e in results.items():
        if "domains" not in e:
            continue
        cfg = e.get("config", {})
        for dom, d in e["domains"].items():
            for it in d["iters"]:
                row = {"key": key, "run": e["run"], "epoch": e["epoch"],
                       "lambda_commute": cfg.get("lambda_commute"), "lr": cfg.get("lr"),
                       "commute_metric": cfg.get("commute_metric"),
                       "domain": dom, "iters": it["n"], "joint": it["joint"],
                       "joint_reverse_order": it["joint_reverse_order"],
                       "order_dist": it["order_dist"], "sharpness_rel": it["sharpness_rel"],
                       "diversity_rel": it.get("diversity_rel"),
                       "kid": it.get("kid"), "fid": it.get("fid"),
                       "joint_residual": it["joint_residual"]}
                for attr, v in it["per_attr"].items():
                    row[f"rate_{attr}"] = v
                for tag in ("single_A", "single_B"):
                    for attr, v in d[tag]["per_attr"].items():
                        row[f"{tag}_{attr}"] = v
                rows.append(row)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(args):
    set_seed(args.seed)
    device = resolve_device(args)
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.json")
    grid_path = os.path.join(args.out, "grid_tensors.pt")
    x0_path = os.path.join(args.out, "x0.pt")

    if os.path.exists(x0_path):
        x0s = {k: v.to(device) for k, v in torch.load(x0_path).items()}
        print(f"reusing x_0 from {x0_path}")
    else:
        x0s = build_x0(args, device)
        torch.save({k: v.cpu() for k, v in x0s.items()}, x0_path)
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    grids = torch.load(grid_path) if os.path.exists(grid_path) else {}
    grids["x0"] = {d: x[:args.grid_n].half().cpu() for d, x in x0s.items()}
    sharp_ref = sharpness(x0s["real"])
    refs = {"sharpness": sharp_ref, "diversity": diversity(x0s["real"])}

    classifier, _ = load_classifier(args.classifier, device)
    spec = args.intersection_spec

    dist_fn = None
    if args.kid:
        extractor = FeatureExtractor(device, fallback_classifier=classifier)
        ds_int = get_filtered_dataset(args, spec, split="test")
        ref_imgs = collect_real_images(ds_int, min(args.kid_ref_n, len(ds_int)), seed=args.seed + 1)
        f_ref = extractor(ref_imgs)

        def dist_fn(imgs):
            f = extractor(imgs.cpu())
            # KID is the headline: unbiased at these sample sizes, where FID
            # (n < 2048 feature dims) is rank-deficient -- kept for reference only
            return {"kid": kernel_distance(f_ref, f),
                    "fid": frechet_distance(f_ref, f, warn_rank=False),
                    "feature_kind": extractor.kind}

    if "reference" not in results or args.force:
        ref = {"sharpness_real": sharp_ref, "diversity_real": refs["diversity"], "spec": spec,
               "x0": {d: attr_rates(classifier, x.cpu(), spec, device) for d, x in x0s.items()}}
        ds_int = get_filtered_dataset(args, spec, split="test")
        real_int = collect_real_images(ds_int, min(args.num_samples, len(ds_int)), seed=args.seed)
        ref["real_intersection"] = attr_rates(classifier, real_int, spec, device)
        if dist_fn:
            ref["real_intersection"].update(dist_fn(real_int))
        if args.baseline:
            base, _ = load_ign(args.baseline, device)
            set_inference_mode([base], eval_mode=False)
            with torch.no_grad():
                outs = {d: base(x) for d, x in x0s.items()}
            ref["baseline"] = {d: {**attr_rates(classifier, o.cpu(), spec, device),
                                   **(dist_fn(o) if dist_fn else {}),
                                   "sharpness_rel": sharpness(o) / sharp_ref,
                                   "diversity_rel": diversity(o) / refs["diversity"]}
                               for d, o in outs.items()}
            grids["baseline"] = {d: o[:args.grid_n].half().cpu() for d, o in outs.items()}
            del base
        results["reference"] = ref
        print(f"reference: ceiling={ref['real_intersection']['joint']:.3f} "
              + (" ".join(f"baseline[{d}]={ref['baseline'][d]['joint']:.3f}"
                          for d in DOMAINS) if args.baseline else ""))

    todo = [("naive", 0, args.naive_a, args.naive_b, {"lambda_commute": None})]
    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(d)})
    for rd in run_dirs:
        cfg = run_config(rd)
        for ep, (pa, pb) in find_snapshots(rd).items():
            if args.epochs and ep not in args.epochs:
                continue
            todo.append((os.path.basename(rd.rstrip("/")), ep, pa, pb, cfg))

    for run, ep, pa, pb, cfg in todo:
        key = f"{run}@{ep}"
        if key in results and key in grids and not args.force:
            continue
        f_a, _ = load_ign(pa, device)
        f_b, _ = load_ign(pb, device)
        doms, g = evaluate_pair(f_a, f_b, x0s, classifier, spec, device,
                                args.max_iters, args.grid_n, refs, dist_fn)
        results[key] = {"run": run, "epoch": ep, "ckpt_a": pa, "ckpt_b": pb,
                        "config": cfg, "domains": doms}
        grids[key] = g
        summary = "  ".join(
            f"{d}: " + "/".join(f"{it['joint']:.2f}" for it in doms[d]["iters"])
            + f" od1={doms[d]['iters'][0]['order_dist']:.1f}"
            + f" sh={doms[d]['iters'][-1]['sharpness_rel']:.2f}"
            + f" div={doms[d]['iters'][-1]['diversity_rel']:.2f}" for d in DOMAINS)
        print(f"{key:40s} {summary}", flush=True)
        del f_a, f_b
        # checkpoint results as we go: a sweep eval can take a while
        with open(res_path, "w") as fh:
            json.dump(results, fh, indent=1)
        torch.save(grids, grid_path)

    with open(res_path, "w") as fh:
        json.dump(results, fh, indent=1)
    torch.save(grids, grid_path)
    write_csv(results, os.path.join(args.out, "results.csv"))
    print(f"wrote {res_path}, results.csv, grid_tensors.pt ({len(results) - 1} entries)")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
