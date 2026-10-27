"""Recompute the evaluation curves for every saved checkpoint of a run and
rebuild loss_curves.png in place.

For each ckpt_*.pt in the run directory (deduplicated by epoch):

  * test recon |f(x)-x| on ONE FIXED batch of 64 has-feature TEST images
    (replaces the old bottom-left panel, which mixed has+lacks images);
  * a FRESH random batch (different per checkpoint) of 64 lacks-feature TEST
    images -> mean classifier P(feature | f(x)): did f add the feature?
  * paired confidence differences on fresh test batches, has and lacks:
    mean[ P(feature | f(x)) - P(feature | x) ] -- for has images this shows
    whether applying f makes the classifier LESS confident about images that
    genuinely have the feature.

No train images are used anywhere. All forwards run with train-mode BatchNorm
(batch statistics), the regime IGNs are meant to run in.

Outputs, inside the run directory:
  ckpt_metrics.json   raw per-checkpoint values
  loss_history.json   updated: old mixed test_recon_l1 dropped, new values in
  loss_curves.png     rebuilt (panels 1-4 unchanged, panel 5 replaced, panel 6
                      = the confidence curves)

Run from the project root (needs ign_* modules, data/, and the classifier):

    python -m experiments.per_network.rebuild_loss_curves --run_dir runs/Smiling_paper500
"""

import argparse
import glob
import json
import os
import types

import torch

from ign.data import ATTR_TO_IDX, CELEBA_ATTRS, get_filtered_dataset, parse_attr_spec
from ign.models import load_classifier, load_ign
from ign.utils import set_seed
from train.train_ign import plot_losses


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
    ap = argparse.ArgumentParser("recompute per-checkpoint test metrics, rebuild curves")
    ap.add_argument("--run_dir", required=True, help="e.g. runs/Smiling_paper500")
    ap.add_argument("--classifier", default="runs/classifier/classifier.pt")
    ap.add_argument("--feature", default=None,
                    help="default: first attribute of the run's attr_spec")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- checkpoints, deduplicated by stored epoch --------------------------
    paths = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    if not paths:
        raise SystemExit(f"no ckpt_*.pt in {args.run_dir}")
    by_epoch = {}
    for p in paths:
        e = torch.load(p, map_location="cpu", weights_only=False)["epoch"]
        by_epoch.setdefault(e, p)
    epochs = sorted(by_epoch)
    print(f"{len(epochs)} checkpoints: epochs {epochs[0]}..{epochs[-1]}")

    # ---- feature, test data, classifier ------------------------------------
    first_ckpt = torch.load(by_epoch[epochs[0]], map_location="cpu", weights_only=False)
    feature = args.feature or CELEBA_ATTRS[
        parse_attr_spec(first_ckpt.get("attr_spec", "Smiling"))[0][0]]
    print(f"feature: {feature}")

    ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
    ds = get_filtered_dataset(ns, "all", split="test")
    pools = load_pools(feature, "test", ds)
    clf, _ = load_classifier(args.classifier, device)
    fi = ATTR_TO_IDX[feature]

    # fixed has-only recon batch, same for every checkpoint
    g0 = torch.Generator().manual_seed(args.seed)
    recon_x = grab(ds, pools["has"], args.batch, g0).to(device)

    # ---- sweep --------------------------------------------------------------
    @torch.no_grad()
    def p_feat(imgs):
        return torch.sigmoid(clf(imgs))[:, fi]

    metrics = []
    for e in epochs:
        model, _ = load_ign(by_epoch[e], device)
        model.train()  # batch-stats BN; model is discarded afterwards
        with torch.no_grad():
            recon = float((model(recon_x) - recon_x).abs().mean())
            gE = torch.Generator().manual_seed(args.seed + 1 + e)  # fresh per ckpt
            x_has = grab(ds, pools["has"], args.batch, gE).to(device)
            x_lacks = grab(ds, pools["lacks"], args.batch, gE).to(device)
            dp_has = float((p_feat(model(x_has)) - p_feat(x_has)).mean())
            p_lacks_after = p_feat(model(x_lacks))
            dp_lacks = float((p_lacks_after - p_feat(x_lacks)).mean())
        row = {"epoch": e, "test_recon_l1": recon,
               "clf_p_after_lacks": float(p_lacks_after.mean()),
               "clf_dp_has": dp_has, "clf_dp_lacks": dp_lacks}
        metrics.append(row)
        print(f"  e{e:04d} recon={recon:.4f} P_after_lacks={row['clf_p_after_lacks']:.3f} "
              f"dP_has={dp_has:+.3f} dP_lacks={dp_lacks:+.3f}")
        del model

    with open(os.path.join(args.run_dir, "ckpt_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ---- merge into loss_history and redraw the figure ----------------------
    hist_path = os.path.join(args.run_dir, "loss_history.json")
    history = []
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            history = json.load(fh)
    for h in history:                      # drop the old mixed-batch values
        h.pop("test_recon_l1", None)
    by_e = {h["epoch"]: h for h in history}
    for row in metrics:
        by_e.setdefault(row["epoch"], {"epoch": row["epoch"]}).update(row)
    history = [by_e[e] for e in sorted(by_e)]
    with open(hist_path, "w") as fh:
        json.dump(history, fh)

    plot_losses(history, args.run_dir)
    print(f"wrote {os.path.join(args.run_dir, 'ckpt_metrics.json')}, "
          f"updated loss_history.json and loss_curves.png")


if __name__ == "__main__":
    main()
