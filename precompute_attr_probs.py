"""One-off sweep: classifier probabilities for ALL 40 attributes over a split.

The attribute classifier predicts every CelebA attribute in one forward pass,
so a single sweep per split yields an (N, 40) probability matrix. Saved once
(~15 MB for both splits), it answers any (split, feature, has/hasn't) query
instantly -- no re-classification, no copying images anywhere. Index i in the
matrix is position i of the full split in dataset order, directly usable with
get_filtered_dataset(..., 'all', split)[i].

Precompute (a few minutes, dominated by JPEG decoding):

    python precompute_attr_probs.py --classifier runs/classifier/classifier.pt

Query from any script:

    from precompute_attr_probs import get_indices
    idx = get_indices("test", "Smiling", has=False)   # most confident first
    ds = get_filtered_dataset(ns, "all", split="test")
    imgs = torch.stack([ds[i][0] for i in idx[:64].tolist()])
"""

import argparse
import os
import time
import types

import torch
from torch.utils.data import DataLoader

from ign_data import ATTR_TO_IDX, CELEBA_ATTRS, get_filtered_dataset
from ign_models import load_classifier

DEFAULT_OUT_DIR = os.path.join("runs", "classifier")


@torch.no_grad()
def precompute_split(split, clf, args, device):
    ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
    ds = get_filtered_dataset(ns, "all", split=split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    t0 = time.time()
    probs = []
    for i, (imgs, _) in enumerate(loader):
        probs.append(torch.sigmoid(clf(imgs.to(device, non_blocking=True))).half().cpu())
        if i % 100 == 0:
            print(f"  [{split}] batch {i}/{len(loader)}", end="\r")
    probs = torch.cat(probs)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"attr_probs_{split}.pt")
    torch.save({"probs": probs, "attrs": list(CELEBA_ATTRS), "split": split}, out)
    print(f"[{split}] {len(probs)} images in {time.time() - t0:.0f}s -> {out}")


def get_indices(split, feature, has, thresh=0.5, out_dir=DEFAULT_OUT_DIR):
    """Indices (into the full split, dataset order) of images the classifier
    says have (has=True) or lack (has=False) the feature, most confident first.
    Loads the precomputed matrix; run this file first if it does not exist.
    """
    path = os.path.join(out_dir, f"attr_probs_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run precompute_attr_probs.py first")
    p = torch.load(path)["probs"][:, ATTR_TO_IDX[feature]].float()
    mask = (p >= thresh) if has else (p < thresh)
    idx = torch.nonzero(mask, as_tuple=True)[0]
    confidence = p[idx] if has else 1 - p[idx]
    return idx[confidence.argsort(descending=True)]


def main():
    ap = argparse.ArgumentParser("precompute classifier probabilities for all attributes")
    ap.add_argument("--classifier", default="runs/classifier/classifier.pt")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf, _ = load_classifier(args.classifier, device)
    for split in args.splits:
        precompute_split(split, clf, args, device)

    # tiny demo query so the output shows what the mapping gives you
    for has in (True, False):
        idx = get_indices(args.splits[-1], "Smiling", has, out_dir=args.out_dir)
        print(f"demo: {args.splits[-1]} Smiling={has}: {len(idx)} images, "
              f"first indices {idx[:5].tolist()}")


if __name__ == "__main__":
    main()
