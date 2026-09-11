"""Show a feature's IGN applied to images WITH and WITHOUT that feature,
from both the train split and the (unseen) test split.

CelebA has official train/valid/test partitions; the IGNs were trained on
'train', so the test rows show generalization. For each split, a pool of
images is scored with the trained attribute classifier and the 3 most
confident feature-HAVING and 3 most confident feature-FREE images are kept.
The IGN is applied with train-mode/batch-statistics BatchNorm (the regime
IGNs are meant to run in; the shown images are forwarded inside a larger
batch so the statistics are stable).

One grid per split, columns = [3 with feature | 3 without feature]:

    row 1: x
    row 2: f(x)      (with-feature columns should barely change: x is near a
                      fixed point; without-feature columns should gain it)
    row 3: f(f(x))   (should barely differ from row 2 if f is idempotent)

Also prints the classifier's mean P(feature) per group before/after each
application.

Put this at the project root (next to ign_models.py). The classifier lives in
the OLD project's runs/classifier/classifier.pt; from the new project pass
--classifier pointing at it (or copy the file over).

    python apply_to_negatives.py --feature Smiling
    python apply_to_negatives.py --feature Smiling \
        --ckpt runs/Smiling_paper500/ckpt_last.pt \
        --classifier ~/old_project/runs/classifier/classifier.pt
"""

import argparse
import os
import types

import torch
import torch.nn as nn

from ign_data import ATTR_TO_IDX, get_filtered_dataset
from ign_models import load_classifier, load_ign
from ign_utils import save_image_grid, set_seed


def bn_batch_stats(model):
    """Train-mode BN with momentum zeroed: batch statistics, buffers untouched."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0


@torch.no_grad()
def classifier_probs(clf, imgs, attr_idx, device, chunk=256):
    probs = []
    for i in range(0, len(imgs), chunk):
        logits = clf(imgs[i:i + chunk].to(device))
        probs.append(torch.sigmoid(logits)[:, attr_idx].cpu())
    return torch.cat(probs)


@torch.no_grad()
def run_split(split, model, clf, attr_idx, args, device):
    ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
    ds = get_filtered_dataset(ns, "all", split=split)
    g = torch.Generator().manual_seed(args.seed)
    pool_idx = torch.randperm(len(ds), generator=g)[:args.pool].tolist()
    imgs = torch.stack([ds[i][0] for i in pool_idx])

    k = args.per_group
    n_shown = 2 * k
    probs = classifier_probs(clf, imgs, attr_idx, device)
    with_idx = probs.argsort(descending=True)[:k]      # most confident HAVE
    without_idx = probs.argsort()[:k]                  # most confident LACK
    shown = torch.cat([with_idx, without_idx])

    # forward the shown images inside a bigger batch for stable BN statistics
    mask = torch.ones(len(imgs), dtype=torch.bool)
    mask[shown] = False
    filler = torch.nonzero(mask, as_tuple=True)[0][:max(args.batch - n_shown, 0)]
    x = torch.cat([imgs[shown], imgs[filler]]).to(device)

    bn_batch_stats(model)
    rows = [x[:n_shown]]
    p = [probs[shown]]
    cur = x
    for _ in range(args.applications):
        cur = model(cur)
        rows.append(cur[:n_shown])
        p.append(classifier_probs(clf, cur[:n_shown].cpu(), attr_idx, device))

    out = f"ign_{args.feature}_{split}.png"
    print("Saving grid to:", out)
    save_image_grid(torch.cat(rows, 0), out, nrow=n_shown)

    print(f"\n[{split}] grid -> {out}  (cols 1-{k} with {args.feature}, "
          f"cols {k + 1}-{n_shown} without)")
    print(f"  classifier mean P({args.feature}):   with-group   without-group")
    labels = ["x"] + [f"f^{k_}(x)" for k_ in range(1, args.applications + 1)]
    for label, pk in zip(labels, p):
        print(f"    {label:8s}                   {pk[:k].mean():.3f}        {pk[k:].mean():.3f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("IGN on with/without-feature images, train & test splits")
    ap.add_argument("--feature", default="Smiling")
    ap.add_argument("--ckpt", default=None,
                    help="IGN checkpoint; default: first existing of "
                         "runs/<feature>_paper500|_paper1000|<feature>/ckpt_last.pt")
    ap.add_argument("--classifier", default="runs/classifier/classifier.pt")
    ap.add_argument("--pool", type=int, default=1024,
                    help="images per split scored by the classifier")
    ap.add_argument("--batch", type=int, default=64,
                    help="forward-batch size the BN statistics come from")
    ap.add_argument("--applications", type=int, default=2,
                    help="how many times f is applied (= number of rows below x)")
    ap.add_argument("--per_group", type=int, default=3,
                    help="images per group (grid shows per_group with + per_group without)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.feature not in ATTR_TO_IDX:
        raise SystemExit(f"unknown feature {args.feature!r}; valid: {', '.join(ATTR_TO_IDX)}")
    attr_idx = ATTR_TO_IDX[args.feature]

    ckpt_path = args.ckpt
    if ckpt_path is None:
        candidates = [f"runs/{args.feature}_paper500/ckpt_last.pt",
                      f"runs/{args.feature}_paper1000/ckpt_last.pt",
                      f"runs/{args.feature}/ckpt_last.pt"]
        ckpt_path = next((c for c in candidates if os.path.exists(c)), None)
        if ckpt_path is None:
            raise SystemExit(f"no checkpoint found among {candidates}; pass --ckpt")

    model, _ = load_ign(ckpt_path, device)
    clf, _ = load_classifier(args.classifier, device)  # eval mode; its BN stats are clean
    print(f"feature={args.feature}  ign={ckpt_path}  classifier={args.classifier}")

    for split in ("train", "test"):
        run_split(split, model, clf, attr_idx, args, device)


if __name__ == "__main__":
    main()
