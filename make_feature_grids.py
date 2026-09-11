"""Rebuild the 4-subset feature-check grids for EVERY checkpoint of a run.

For each ckpt_*.pt in --run_dir (deduplicated by stored epoch) and the given
feature, renders the same grids train_ign.py writes during training:

    <out>/train_fixed_eNNNN.png   <out>/train_fresh_eNNNN.png
    <out>/test_fixed_eNNNN.png    <out>/test_fresh_eNNNN.png

Grid: columns 1..k have the feature, k+1..2k lack it (ground-truth subsets,
from attr_splits/ lists when present); rows are x, f(x), ..., f^apps(x).
'fixed' batches are identical across all checkpoints (seeded) so they compare
like a time-lapse; 'fresh' batches are re-drawn per checkpoint (seeded by
epoch, so re-runs are reproducible). Forwards use train-mode BatchNorm, with
fixed+fresh sharing one batch per split for stable statistics.

Run from the project root:

    python make_feature_grids.py --run_dir runs/Smiling_paper500
    python make_feature_grids.py --run_dir runs/Male_paper500 --feature Male
"""

import argparse
import glob
import os
import types

import torch

from ign_data import ATTR_TO_IDX, CELEBA_ATTRS, get_filtered_dataset, parse_attr_spec
from ign_models import load_ign
from ign_utils import save_image_grid, set_seed


def load_pools(feature_name, split, ds):
    """has/lacks id pools: saved attr_splits lists if present, else labels."""
    for root in ("attr_splits", "attr_split"):
        d = os.path.join(root, feature_name, split)
        if (os.path.exists(os.path.join(d, "has.txt"))
                and os.path.exists(os.path.join(d, "lacks.txt"))):
            pools = {}
            for name in ("has", "lacks"):
                with open(os.path.join(d, f"{name}.txt")) as fh:
                    pools[name] = torch.tensor(
                        [int(line.split("\t")[0]) for line in fh if line.strip()])
            print(f"[{split}] pools from {d}")
            return pools
    col = ds.base.attr[:, ATTR_TO_IDX[feature_name]]
    print(f"[{split}] no saved lists -- pools computed from labels")
    return {"has": (col == 1).nonzero(as_tuple=True)[0],
            "lacks": (col == 0).nonzero(as_tuple=True)[0]}


def grab(ds, pool, n, gen):
    sel = pool[torch.randperm(len(pool), generator=gen)[:n]]
    return torch.stack([ds[i][0] for i in sel.tolist()])


def main():
    ap = argparse.ArgumentParser("4-subset feature grids for every checkpoint of a run")
    ap.add_argument("--run_dir", required=True, help="e.g. runs/Smiling_paper500")
    ap.add_argument("--feature", default=None,
                    help="default: first attribute of the run's attr_spec")
    ap.add_argument("--per_group", type=int, default=10, help="images per has/lacks group")
    ap.add_argument("--applications", type=int, default=2,
                    help="how many times f is applied (rows below x)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default=None,
                    help="default: <run_dir>/feature_check_sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(args.run_dir, "feature_check_sweep")
    os.makedirs(out_dir, exist_ok=True)

    # checkpoints, deduplicated by stored epoch
    paths = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    if not paths:
        raise SystemExit(f"no ckpt_*.pt in {args.run_dir}")
    by_epoch = {}
    for p in paths:
        e = torch.load(p, map_location="cpu", weights_only=False)["epoch"]
        by_epoch.setdefault(e, p)
    epochs = sorted(by_epoch)
    print(f"{len(epochs)} checkpoints: epochs {epochs[0]}..{epochs[-1]}")

    first = torch.load(by_epoch[epochs[0]], map_location="cpu", weights_only=False)
    feature = args.feature or CELEBA_ATTRS[
        parse_attr_spec(first.get("attr_spec", "Smiling"))[0][0]]
    print(f"feature: {feature}  per_group: {args.per_group}  "
          f"applications: {args.applications}")

    # datasets, pools, and the fixed batches (identical for every checkpoint)
    k = args.per_group
    splits = {}
    g_fixed = torch.Generator().manual_seed(args.seed + 1234)
    for split in ("train", "test"):
        ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
        ds = get_filtered_dataset(ns, "all", split=split)
        pools = load_pools(feature, split, ds)
        fixed = torch.cat([grab(ds, pools["has"], k, g_fixed),
                           grab(ds, pools["lacks"], k, g_fixed)])
        splits[split] = {"ds": ds, "pools": pools, "fixed": fixed.to(device)}

    @torch.no_grad()
    def render(model, epoch):
        model.train()  # batch-stats BN; model is reloaded per ckpt anyway
        for split, s in splits.items():
            gE = torch.Generator().manual_seed(args.seed + 1 + epoch)
            fresh = torch.cat([grab(s["ds"], s["pools"]["has"], k, gE),
                               grab(s["ds"], s["pools"]["lacks"], k, gE)]).to(device)
            x = torch.cat([s["fixed"], fresh])
            rows = {"fixed": [x[:2 * k]], "fresh": [x[2 * k:]]}
            cur = x
            for _ in range(args.applications):
                cur = model(cur)
                rows["fixed"].append(cur[:2 * k])
                rows["fresh"].append(cur[2 * k:])
            for kind, r in rows.items():
                save_image_grid(torch.cat(r, 0),
                                os.path.join(out_dir, f"{split}_{kind}_e{epoch:04d}.png"),
                                nrow=2 * k)

    for e in epochs:
        model, _ = load_ign(by_epoch[e], device)
        render(model, e)
        print(f"  e{e:04d} done")
        del model

    print(f"grids -> {out_dir}")


if __name__ == "__main__":
    main()
