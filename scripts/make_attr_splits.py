"""Write, for each feature, train/test lists of images that have / lack it,
using the dataset's ground-truth labels (no classifier, no image copying).

Output layout (one directory per feature, one per split, two lists each):

    attr_splits/
      Smiling/
        train/has.txt      lines: "<index>\t<filename>"
        train/lacks.txt
        test/has.txt
        test/lacks.txt
      Eyeglasses/
        ...

<index> is the image's position in the FULL split in dataset order -- directly
usable as get_filtered_dataset(ns, "all", split)[index]. <filename> is the jpg
name under data/celeba/img_align_celeba/. Only labels are read, no image is
decoded, so the whole thing takes seconds.

    python -m scripts.make_attr_splits                      # all 40 features
    python -m scripts.make_attr_splits --features Smiling Male Eyeglasses

Read a list back from any script:

    from scripts.make_attr_splits import load_ids
    ids, files = load_ids("Smiling", "test", has=False)
"""

import argparse
import os
import types

from ign.data import CELEBA_ATTRS, get_filtered_dataset

DEFAULT_OUT_DIR = "attr_splits"


def load_ids(feature, split, has, out_dir=DEFAULT_OUT_DIR):
    """Returns (indices, filenames) from a previously written list."""
    path = os.path.join(out_dir, feature, split, "has.txt" if has else "lacks.txt")
    ids, files = [], []
    with open(path) as fh:
        for line in fh:
            i, name = line.rstrip("\n").split("\t")
            ids.append(int(i))
            files.append(name)
    return ids, files


def main():
    ap = argparse.ArgumentParser("ground-truth has/lacks lists per feature and split")
    ap.add_argument("--features", nargs="+", default=None,
                    help="default: all 40 CelebA attributes")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    features = args.features or list(CELEBA_ATTRS)
    unknown = [f for f in features if f not in CELEBA_ATTRS]
    if unknown:
        raise SystemExit(f"unknown features {unknown}; valid: {', '.join(CELEBA_ATTRS)}")

    for split in args.splits:
        ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
        base = get_filtered_dataset(ns, "all", split=split).base
        attr = base.attr          # (N, 40) ground-truth 0/1
        names = base.filename     # jpg filenames, same order
        for feat in features:
            col = attr[:, CELEBA_ATTRS.index(feat)]
            for fname, keep in (("has.txt", col == 1), ("lacks.txt", col == 0)):
                d = os.path.join(args.out_dir, feat, split)
                os.makedirs(d, exist_ok=True)
                idx = keep.nonzero(as_tuple=True)[0].tolist()
                with open(os.path.join(d, fname), "w") as fh:
                    for i in idx:
                        fh.write(f"{i}\t{names[i]}\n")
                print(f"{split:5s} {feat:20s} {fname:9s} {len(idx):7d} images")


if __name__ == "__main__":
    main()
