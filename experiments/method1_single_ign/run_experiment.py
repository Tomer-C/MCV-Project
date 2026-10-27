"""The main experiment: alternating projection vs. the trained intersection IGN.

Runs the four phases of the proposal's Method section end to end and writes a
JSON results file plus figures.

  Phase 2  Alternating projection    x_{n+1} = (f_B o f_A)(x_n), tracked
  Phase 4  Comparison                x* vs f_AnB(x_0), on the same x_0
           Order sensitivity         every permutation of the models
           Scaling                   works unchanged for K > 2 models

Example
-------
  python -m experiments.method1_single_ign.run_experiment \
      --models runs/Smiling/ckpt_last.pt runs/Eyeglasses/ckpt_last.pt \
      --names f_A f_B \
      --baseline runs/Smiling_AND_Eyeglasses/ckpt_last.pt \
      --intersection_spec "Smiling,Eyeglasses" \
      --classifier runs/classifier/classifier.pt \
      --num_iters 20 --num_samples 512
"""

import argparse
import json
import os

import torch

from ign.alternating import (alternating_projection, damped_alternating_projection,
                         joint_fixed_point_residual, order_disagreement,
                         run_all_orders, set_inference_mode)
from ign.data import get_filtered_dataset, spec_to_slug
from ign.models import load_classifier, load_ign
from ign.utils import resolve_device, sample_freq_noise, save_image_grid, set_seed
from ign.metrics import (FeatureExtractor, attr_rates, collect_real_images,
                     frechet_distance, kernel_distance, paired_distances)


def get_args_parser():
    p = argparse.ArgumentParser("Alternating projection experiment")
    p.add_argument("--models", nargs="+", required=True,
                   help="checkpoints for f_A, f_B, ... in application order")
    p.add_argument("--names", nargs="+", default=None, help="labels, e.g. f_A f_B")
    p.add_argument("--baseline", default=None,
                   help="checkpoint for f_AnB trained directly on the intersection")
    p.add_argument("--intersection_spec", required=True,
                   help="attribute spec defining the intersection, e.g. 'Smiling,Eyeglasses'")
    p.add_argument("--classifier", default="runs/classifier/classifier.pt")

    p.add_argument("--num_iters", default=20, type=int)
    p.add_argument("--num_samples", default=512, type=int)
    p.add_argument("--init", default="noise", choices=["noise", "white", "real"],
                   help="x_0: 'noise' = frequency-matched noise using the stats "
                        "saved in the first model's checkpoint (paper-style; "
                        "falls back to white if absent), 'white' = white "
                        "gaussian, 'real' = real test images")
    p.add_argument("--init_spec", default="all",
                   help="when --init real, which subset x_0 is drawn from")
    p.add_argument("--noise_stats_spec", default="",
                   help="with --init noise: build the FFT noise statistics from "
                        "a train batch matching this attr spec (e.g. the target "
                        "intersection 'Male,Smiling') instead of the first "
                        "checkpoint's stored stats")
    p.add_argument("--bn_mode", default="train", choices=["train", "eval"],
                   help="train = batch-statistics BatchNorm, the only mode "
                        "upstream IGN ever runs in (running stats untouched); "
                        "eval = running stats, deterministic but polluted by "
                        "the noise batches seen in training")
    p.add_argument("--damped_alpha", default=0.5, type=float)
    p.add_argument("--skip_damped", action="store_true")
    p.add_argument("--skip_orders", action="store_true")
    p.add_argument("--skip_fid", action="store_true")

    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--tag", default=None)
    return p


@torch.no_grad()
def apply_in_batches(model, x, bs, eval_mode=False):
    """No-grad forward in mini-batches (chunked to avoid OOM on large x).

    BatchNorm behaviour follows `set_inference_mode`, matching whatever the
    alternating projection uses; in batch-stats mode the statistics come from
    each chunk of `bs` images.
    """
    set_inference_mode([model], eval_mode)
    return torch.cat([model(x[i:i + bs]) for i in range(0, x.shape[0], bs)], dim=0)


def main(args):
    set_seed(args.seed)
    device = resolve_device(args)
    names = args.names or [f"f{i}" for i in range(len(args.models))]
    assert len(names) == len(args.models), "--names must match --models in length"

    tag = args.tag or spec_to_slug(args.intersection_spec)
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    # ---- load the per-feature IGNs -------------------------------------
    models, model_specs, model_ckpts = [], [], []
    for path, name in zip(args.models, names):
        m, ckpt = load_ign(path, device)
        models.append(m)
        model_specs.append(ckpt.get("attr_spec", "?"))
        model_ckpts.append(ckpt)
        print(f"loaded {name}: {path} (trained on {model_specs[-1]!r})")

    baseline = None
    if args.baseline:
        baseline, bckpt = load_ign(args.baseline, device)
        print(f"loaded baseline f_AnB: {args.baseline} (trained on {bckpt.get('attr_spec','?')!r})")

    classifier = None
    if args.classifier and os.path.exists(args.classifier):
        classifier, cckpt = load_classifier(args.classifier, device)
        print(f"loaded classifier (val_acc={cckpt.get('val_acc', float('nan')):.4f})")
    else:
        print("WARNING: no classifier found; attribute-satisfaction metrics will be skipped.")

    results = {"config": vars(args), "model_specs": dict(zip(names, model_specs))}

    # ---- x_0 ------------------------------------------------------------
    if args.init == "noise" and args.noise_stats_spec:
        # FFT statistics of a chosen subset (e.g. the target intersection)
        # instead of the stats stored in the first checkpoint
        from ign.utils import freq_stats_from_images
        ds_stats = get_filtered_dataset(args, args.noise_stats_spec, split="train")
        g = torch.Generator().manual_seed(args.seed)
        sel = torch.randperm(len(ds_stats), generator=g)[:256]
        imgs = torch.stack([ds_stats[i][0] for i in sel.tolist()]).to(device)
        x0 = sample_freq_noise(freq_stats_from_images(imgs), args.num_samples)
        init_desc = f"fft-matched noise (stats of {args.noise_stats_spec!r} train images)"
    elif args.init in ("noise", "white"):
        freq_stats = model_ckpts[0].get("freq_stats") if args.init == "noise" else None
        if freq_stats is not None:
            freq_stats = {k: v.to(device) for k, v in freq_stats.items()}
            x0 = sample_freq_noise(freq_stats, args.num_samples)
            init_desc = "fft-matched noise (stats from checkpoint)"
        else:
            if args.init == "noise":
                print("WARNING: checkpoint carries no freq_stats (trained before "
                      "the paper-faithful noise change); using white noise.")
            x0 = torch.randn(args.num_samples, 3, args.image_size, args.image_size,
                             device=device)
            init_desc = "white noise"
    else:
        ds = get_filtered_dataset(args, args.init_spec, split="test")
        x0 = collect_real_images(ds, args.num_samples, seed=args.seed).to(device)
        init_desc = f"real images ({args.init_spec})"
    print(f"x_0: {tuple(x0.shape)} from {init_desc}")

    # =====================================================================
    # Phase 2: alternating projection
    # =====================================================================
    print("\n=== Phase 2: alternating projection ===")
    eval_mode = args.bn_mode == "eval"
    alt = alternating_projection(models, x0, num_iters=args.num_iters,
                                 track_trajectory=True, trajectory_every=1,
                                 eval_mode=eval_mode)
    x_star = alt["final"]
    m = alt["metrics"]

    results["alternating"] = {
        "step_norm_mean": m["step_norm"].mean(1).tolist(),
        "step_norm_final": float(m["step_norm"][-1].mean()),
        "image_norm_mean": m["image_norm"].mean(1).tolist(),
        "cycle_distance_mean": m["cycle_distance"].mean(1).tolist(),
        **{f"idem_residual_{names[i]}": m[f"idem_residual_{i}"].mean(1).tolist()
           for i in range(len(models))},
        "joint_residual_final": float(joint_fixed_point_residual(models, x_star).mean()),
        "converged": bool(m["step_norm"][-1].mean() < 0.05 * m["step_norm"][0].mean()),
        "diverged": bool(m["image_norm"][-1].mean() > 10 * m["image_norm"][0].mean()),
    }
    print(f"  step norm: {m['step_norm'][0].mean():.4f} -> {m['step_norm'][-1].mean():.4f}")
    print(f"  joint fixed-point residual: {results['alternating']['joint_residual_final']:.4f}")
    print(f"  converged={results['alternating']['converged']} "
          f"diverged={results['alternating']['diverged']}")

    save_image_grid(torch.cat([t[:8] for t in alt["trajectory"][:12]], 0),
                    os.path.join(fig_dir, "trajectory_grid.png"), nrow=8)

    # damped variant, the standard fix if the undamped iteration oscillates
    if not args.skip_damped:
        damped = damped_alternating_projection(models, x0, num_iters=args.num_iters,
                                               alpha=args.damped_alpha, eval_mode=eval_mode)
        results["damped"] = {
            "alpha": args.damped_alpha,
            "step_norm_mean": damped["metrics"]["step_norm"].mean(1).tolist(),
            "joint_residual_mean": damped["metrics"]["joint_residual"].mean(1).tolist(),
            "joint_residual_final": float(damped["metrics"]["joint_residual"][-1].mean()),
        }
        print(f"  damped(a={args.damped_alpha}) joint residual: "
              f"{results['damped']['joint_residual_final']:.4f}")

    # =====================================================================
    # Attribute satisfaction: does x* have all the features?
    # =====================================================================
    if classifier is not None:
        print("\n=== Attribute satisfaction ===")
        results["attr_rates"] = {}
        rate_x0 = attr_rates(classifier, x0.cpu(), args.intersection_spec, device)
        results["attr_rates"]["x0"] = rate_x0
        print(f"  x_0                joint={rate_x0['joint']:.3f}")

        rate_alt = attr_rates(classifier, x_star.cpu(), args.intersection_spec, device)
        results["attr_rates"]["alternating"] = rate_alt
        print(f"  alternating x*     joint={rate_alt['joint']:.3f}  {rate_alt['per_attr']}")

        # each single IGN alone, to show the alternation is doing something
        for i, (mdl, nm) in enumerate(zip(models, names)):
            single = apply_in_batches(mdl, x0, args.batch_size, eval_mode)
            r = attr_rates(classifier, single.cpu(), args.intersection_spec, device)
            results["attr_rates"][f"single_{nm}"] = r
            print(f"  {nm}(x_0) alone      joint={r['joint']:.3f}  {r['per_attr']}")

        if baseline is not None:
            base_out = apply_in_batches(baseline, x0, args.batch_size, eval_mode)
            r = attr_rates(classifier, base_out.cpu(), args.intersection_spec, device)
            results["attr_rates"]["baseline"] = r
            print(f"  baseline f_AnB     joint={r['joint']:.3f}  {r['per_attr']}")

        ds_int = get_filtered_dataset(args, args.intersection_spec, split="test")
        real_int = collect_real_images(ds_int, min(args.num_samples, len(ds_int)), seed=args.seed)
        r = attr_rates(classifier, real_int, args.intersection_spec, device)
        results["attr_rates"]["real_intersection"] = r
        print(f"  real A-and-B       joint={r['joint']:.3f}   (classifier ceiling)")

    # =====================================================================
    # Phase 4: head-to-head against the trained baseline
    # =====================================================================
    if baseline is not None:
        print("\n=== Phase 4: alternating vs trained baseline ===")
        base_out = apply_in_batches(baseline, x0, args.batch_size, eval_mode)
        results["paired_vs_baseline"] = paired_distances(x_star.cpu(), base_out.cpu(), device)
        print(f"  {results['paired_vs_baseline']}")
        save_image_grid(torch.cat([x0[:8], x_star[:8], base_out[:8]], 0),
                        os.path.join(fig_dir, "alt_vs_baseline.png"), nrow=8)

    # =====================================================================
    # Distribution match against real intersection images
    # =====================================================================
    if not args.skip_fid:
        print("\n=== Distribution match (vs real A-and-B images) ===")
        ds_int = get_filtered_dataset(args, args.intersection_spec, split="test")
        n_ref = min(args.num_samples, len(ds_int))
        real_int = collect_real_images(ds_int, n_ref, seed=args.seed)

        extractor = FeatureExtractor(device, fallback_classifier=classifier)
        f_real = extractor(real_int)
        entries = {"alternating": x_star.cpu()}
        if baseline is not None:
            entries["baseline"] = apply_in_batches(baseline, x0, args.batch_size, eval_mode).cpu()
        for i, nm in enumerate(names):
            entries[f"single_{nm}"] = apply_in_batches(models[i], x0, args.batch_size, eval_mode).cpu()

        results["distribution"] = {"feature_kind": extractor.kind, "n_reference": n_ref}
        for key, imgs in entries.items():
            feats = extractor(imgs)
            results["distribution"][key] = {
                "fid": frechet_distance(f_real, feats),
                "kid": kernel_distance(f_real, feats),
            }
            print(f"  {key:16s} FID={results['distribution'][key]['fid']:.2f} "
                  f"KID={results['distribution'][key]['kid']:.4f}")

    # =====================================================================
    # Order sensitivity
    # =====================================================================
    if not args.skip_orders and len(models) > 1:
        print("\n=== Order sensitivity ===")
        orders = run_all_orders(models, x0, num_iters=args.num_iters, names=names,
                                eval_mode=eval_mode)
        results["orders"] = {}
        for key, res in orders.items():
            entry = {
                "step_norm_final": float(res["metrics"]["step_norm"][-1].mean()),
                "joint_residual_final": float(joint_fixed_point_residual(models, res["final"]).mean()),
            }
            if classifier is not None:
                entry["joint_attr_rate"] = attr_rates(
                    classifier, res["final"].cpu(), args.intersection_spec, device)["joint"]
            results["orders"][key] = entry
            print(f"  {key:24s} residual={entry['joint_residual_final']:.4f} "
                  f"attr={entry.get('joint_attr_rate', float('nan')):.3f}")
        results["order_disagreement"] = order_disagreement(orders)
        print("  pairwise limit distances:")
        for k, v in results["order_disagreement"].items():
            print(f"    {k}: {v:.4f}")

    # ---- save ------------------------------------------------------------
    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_json}")

    torch.save({"x0": x0.cpu(), "x_star": x_star.cpu(),
                "trajectory": [t.cpu() for t in alt["trajectory"]],
                "metrics": {k: v for k, v in alt["metrics"].items()}},
               os.path.join(out_dir, "tensors.pt"))
    print(f"wrote {os.path.join(out_dir, 'tensors.pt')}")
    return results


if __name__ == "__main__":
    main(get_args_parser().parse_args())
