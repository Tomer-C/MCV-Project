"""Lay out the Hugging Face CelebA mirror (flwrlabs/celeba) as data/celeba/.

Use when torchvision's Google-Drive download is rate-limited. The mirror keeps
the official train/valid/test partition and the 40 attributes; it ships as
parquet, so this writes the jpgs plus the five list files torchvision reads.
Bounding boxes and landmarks are not in the mirror and are written as zeros
(nothing in this repo uses them).

  python scripts/celeba_from_hf.py data/hf_celeba/test-*.parquet --out data
"""

import argparse
import glob
import io
import os

import pyarrow.parquet as pq
from PIL import Image

from ign.data import CELEBA_ATTRS

SPLIT_ID = {"train": 0, "valid": 1, "test": 2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquets", nargs="+")
    p.add_argument("--out", default="data")
    args = p.parse_args()
    base = os.path.join(args.out, "celeba")
    img_dir = os.path.join(base, "img_align_celeba")
    os.makedirs(img_dir, exist_ok=True)

    rows = []  # (filename, split_id, celeb_id, attrs)
    files = sorted(f for pat in args.parquets for f in glob.glob(pat))
    for path in files:
        split = os.path.basename(path).split("-")[0]
        table = pq.read_table(path)
        cols = table.to_pydict()
        for i, img in enumerate(cols["image"]):
            name = os.path.basename(img.get("path") or "") or f"{split}_{len(rows):06d}.jpg"
            out = os.path.join(img_dir, name)
            if not os.path.exists(out):
                im = Image.open(io.BytesIO(img["bytes"]))
                if name.lower().endswith((".jpg", ".jpeg")) and im.format == "JPEG":
                    with open(out, "wb") as fh:
                        fh.write(img["bytes"])
                else:
                    im.convert("RGB").save(out, quality=95)
            rows.append((name, SPLIT_ID[split], cols["celeb_id"][i],
                         [1 if cols[a][i] else -1 for a in CELEBA_ATTRS]))
        print(f"{path}: {table.num_rows} images")

    def write(fname, lines):
        with open(os.path.join(base, fname), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    n = str(len(rows))
    write("list_eval_partition.txt", [f"{r[0]} {r[1]}" for r in rows])
    write("identity_CelebA.txt", [f"{r[0]} {r[2]}" for r in rows])
    write("list_attr_celeba.txt", [n, " ".join(CELEBA_ATTRS)]
          + [f"{r[0]} " + " ".join(map(str, r[3])) for r in rows])
    write("list_bbox_celeba.txt", [n, "image_id x_1 y_1 width height"]
          + [f"{r[0]} 0 0 0 0" for r in rows])
    write("list_landmarks_align_celeba.txt",
          [n, "lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y "
              "leftmouth_x leftmouth_y rightmouth_x rightmouth_y"]
          + [f"{r[0]} " + " ".join(["0"] * 10) for r in rows])
    print(f"wrote {len(rows)} images -> {base}")


if __name__ == "__main__":
    main()
