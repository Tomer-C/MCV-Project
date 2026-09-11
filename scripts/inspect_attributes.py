"""Count CelebA attribute subsets and their intersections.

Run this FIRST. The proposal flags data scarcity for the A-and-B baseline as
the main feasibility risk; this script tells you which feature pairs actually
have enough overlapping images to train f_AnB on, before you spend GPU hours.

  python scripts/inspect_attributes.py
  python scripts/inspect_attributes.py --specs Smiling Eyeglasses Male Blond_Hair
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ign_data import CELEBA_ATTRS, _load_celeba, get_attr_indices

# Pairs worth considering: both attributes common, and not mutually exclusive.
SUGGESTED = [
    ("Smiling", "Eyeglasses"),
    ("Smiling", "Male"),
    ("Male", "Black_Hair"),
    ("Smiling", "Heavy_Makeup"),
    ("Male", "Smiling"),
    ("Young", "Eyeglasses"),
    ("Blond_Hair", "Smiling"),
    ("Wearing_Hat", "Smiling"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_size", default=64, type=int)
    p.add_argument("--split", default="train")
    p.add_argument("--specs", nargs="*", default=None,
                   help="attributes to cross-tabulate; defaults to a curated list")
    p.add_argument("--min_images", default=3000, type=int,
                   help="threshold below which a baseline subset is called risky")
    args = p.parse_args()

    base = _load_celeba(args.data_root, args.split, args.image_size)
    total = len(base)
    print(f"CelebA {args.split}: {total} images\n")

    if args.specs:
        attrs = args.specs
    else:
        attrs = sorted({a for pair in SUGGESTED for a in pair})

    print("individual attribute counts")
    print("-" * 52)
    idx_sets = {}
    for a in attrs:
        idx = set(get_attr_indices(base, a))
        idx_sets[a] = idx
        print(f"  {a:20s} {len(idx):7d}  ({100*len(idx)/total:5.1f}%)")

    print("\npairwise intersections")
    print("-" * 52)
    rows = []
    for a, b in itertools.combinations(attrs, 2):
        n = len(idx_sets[a] & idx_sets[b])
        rows.append((n, a, b))
    for n, a, b in sorted(rows, reverse=True):
        flag = "" if n >= args.min_images else "  <-- risky for baseline"
        print(f"  {a:18s} & {b:18s} {n:7d}  ({100*n/total:5.1f}%){flag}")

    print("\ntriple intersections (for the K>2 scaling experiment)")
    print("-" * 52)
    triples = []
    for a, b, c in itertools.combinations(attrs, 3):
        n = len(idx_sets[a] & idx_sets[b] & idx_sets[c])
        triples.append((n, a, b, c))
    for n, a, b, c in sorted(triples, reverse=True)[:12]:
        flag = "" if n >= args.min_images else "  <-- risky"
        print(f"  {a:16s} & {b:16s} & {c:16s} {n:6d}{flag}")

    print(f"\nRule of thumb: aim for >= {args.min_images} images in A-and-B so f_AnB "
          "trains to a comparable quality as f_A and f_B.")


if __name__ == "__main__":
    main()
