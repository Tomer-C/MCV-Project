#!/usr/bin/env python
"""Arrange a manually-downloaded CelebA into the layout torchvision expects.

torchvision's CelebA loader wants five whitespace ``.txt`` metadata files plus
an ``img_align_celeba/`` folder under ``<data_root>/celeba/``. The Kaggle mirror
(jessicali9530/celeba-dataset) ships the metadata as ``.csv`` with different
formatting and nests the images one level deep, so torchvision can't read it
directly. This script converts/relocates everything into the right shape.

Usage:
    python scripts/setup_celeba.py --src <kaggle_download_dir> --data_root ./data

It is idempotent and accepts either .csv (Kaggle) or already-correct .txt files.
If identity data is missing (the Kaggle set omits it, and this project never
uses it) a placeholder is synthesised so the loader is satisfied.
"""
import argparse
import csv
import os
import shutil


def _find(src, *names):
    """Return the first existing path among candidate names, searching one level deep."""
    for n in names:
        p = os.path.join(src, n)
        if os.path.exists(p):
            return p
    for root, _dirs, files in os.walk(src):
        for n in names:
            if n in files:
                return os.path.join(root, n)
    return None


def _read_csv(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return rows[0], rows[1:]  # header, body


def _write_headered(dst, header_cols, body, cast_int_from=1):
    """Write a torchvision ``header=1`` file: count line, header line, then rows."""
    with open(dst, "w") as f:
        f.write(f"{len(body)}\n")
        f.write(" ".join(header_cols) + "\n")
        for r in body:
            vals = [str(int(float(v))) for v in r[cast_int_from:]]
            f.write(r[0] + " " + " ".join(vals) + "\n")


def _write_plain(dst, body):
    """Write a torchvision ``header=None`` file: ``<filename> <int>`` per line."""
    with open(dst, "w") as f:
        for r in body:
            f.write(r[0] + " " + " ".join(str(int(float(v))) for v in r[1:]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing the downloaded CelebA files")
    ap.add_argument("--data_root", default="./data")
    args = ap.parse_args()

    dst = os.path.join(args.data_root, "celeba")
    os.makedirs(dst, exist_ok=True)

    # --- images: find the folder of jpgs (Kaggle nests it) and place at celeba/img_align_celeba ---
    img_dst = os.path.join(dst, "img_align_celeba")
    if not (os.path.isdir(img_dst) and any(f.endswith(".jpg") for f in os.listdir(img_dst))):
        img_src = None
        for root, dirs, files in os.walk(args.src):
            if any(f.endswith(".jpg") for f in files):
                img_src = root
                break
        if img_src is None:
            raise SystemExit(f"No .jpg images found under {args.src}")
        if os.path.abspath(img_src) != os.path.abspath(img_dst):
            os.makedirs(img_dst, exist_ok=True)
            n = 0
            for f in os.listdir(img_src):
                if f.endswith(".jpg"):
                    shutil.move(os.path.join(img_src, f), os.path.join(img_dst, f))
                    n += 1
            print(f"[images] moved {n} jpgs -> {img_dst}")
    print(f"[images] ok: {img_dst} ({sum(f.endswith('.jpg') for f in os.listdir(img_dst))} jpgs)")

    # --- attributes (header=1, values -1/1) ---
    attr = _find(args.src, "list_attr_celeba.csv", "list_attr_celeba.txt")
    if attr is None:
        raise SystemExit("list_attr_celeba.{csv,txt} not found in --src")
    header, body = _read_csv(attr) if attr.endswith(".csv") else _read_txt_headered(attr)
    _write_headered(os.path.join(dst, "list_attr_celeba.txt"), header[1:], body)
    filenames = [r[0] for r in body]
    print(f"[attr] wrote list_attr_celeba.txt ({len(body)} rows, {len(header)-1} attrs)")

    # --- partition (header=None) ---
    part = _find(args.src, "list_eval_partition.csv", "list_eval_partition.txt")
    if part is None:
        raise SystemExit("list_eval_partition.{csv,txt} not found in --src")
    _, pbody = _read_csv(part) if part.endswith(".csv") else ([], _read_txt_plain(part))
    _write_plain(os.path.join(dst, "list_eval_partition.txt"), pbody)
    print(f"[partition] wrote list_eval_partition.txt ({len(pbody)} rows)")

    # --- bbox (header=1) ---
    bbox = _find(args.src, "list_bbox_celeba.csv", "list_bbox_celeba.txt")
    if bbox and bbox.endswith(".csv"):
        h, b = _read_csv(bbox)
        _write_headered(os.path.join(dst, "list_bbox_celeba.txt"), h, b)
        print(f"[bbox] wrote list_bbox_celeba.txt ({len(b)} rows)")
    elif bbox:
        shutil.copy(bbox, os.path.join(dst, "list_bbox_celeba.txt"))

    # --- landmarks (header=1) ---
    lm = _find(args.src, "list_landmarks_align_celeba.csv", "list_landmarks_align_celeba.txt")
    if lm and lm.endswith(".csv"):
        h, b = _read_csv(lm)
        _write_headered(os.path.join(dst, "list_landmarks_align_celeba.txt"), h, b)
        print(f"[landmarks] wrote list_landmarks_align_celeba.txt ({len(b)} rows)")
    elif lm:
        shutil.copy(lm, os.path.join(dst, "list_landmarks_align_celeba.txt"))

    # --- identity (header=None); synthesise if absent (Kaggle omits it; project never uses it) ---
    ident = _find(args.src, "identity_CelebA.txt", "identity_CelebA.csv")
    ident_dst = os.path.join(dst, "identity_CelebA.txt")
    if ident and ident.endswith(".csv"):
        _, ib = _read_csv(ident)
        _write_plain(ident_dst, ib)
        print(f"[identity] converted identity_CelebA.txt ({len(ib)} rows)")
    elif ident:
        shutil.copy(ident, ident_dst)
        print("[identity] copied identity_CelebA.txt")
    else:
        with open(ident_dst, "w") as f:
            for fn in filenames:
                f.write(f"{fn} 1\n")
        print(f"[identity] synthesised placeholder identity_CelebA.txt ({len(filenames)} rows)")

    print("\nDone. Now the loader must skip torchvision's md5 gate -- apply the ign_data.py")
    print("patch (see instructions) or your first run will try to re-download from Drive.")


def _read_txt_headered(path):
    with open(path) as f:
        lines = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    header = ["image_id"] + lines[1]
    body = lines[2:]
    return header, body


def _read_txt_plain(path):
    with open(path) as f:
        return [ln.split() for ln in f.read().splitlines() if ln.strip()]


if __name__ == "__main__":
    main()
