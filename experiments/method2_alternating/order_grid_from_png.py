"""Order-of-application grid on the CelebA faces already in the repo.

Cuts the input row out of a saved results grid (save_image_grid layout:
64px tiles, 2px padding) -- by default the 8 non-Male, non-Smiling CelebA
test faces in results/method2_alternating/limits/real_image_editing_grid.png -- and runs both
orders on them, writing the same labeled grid as experiments/method2_alternating/make_order_figures.py.
Qualitative only: 8 faces, and BatchNorm statistics come from these 8.

  python -m experiments.method2_alternating.order_grid_from_png --tag early
  python -m experiments.method2_alternating.order_grid_from_png --tag late \
      --ckpt Male=Male_500_withsamples/ckpt_0300.pt Smiling=Smiling_paper500/ckpt_0325.pt
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from ign.alternating import set_inference_mode
from ign.models import load_ign
from experiments.method2_alternating.make_order_figures import grid
from experiments.method2_alternating.order_experiment import BASELINES, CKPTS, apply, run_order


def tiles_from_grid(path, row=0, n=8, size=64, pad=2):
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 127.5 - 1
    y = pad + row * (size + pad)
    xs = [img[y:y + size, pad + c * (size + pad):pad + c * (size + pad) + size] for c in range(n)]
    return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--png", default=os.path.join("results", "method2_alternating", "limits",
                                                 "real_image_editing_grid.png"))
    p.add_argument("--desc", default="the repo's 8 CelebA test faces with neither Male nor Smiling")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--ckpt", nargs="+", default=[])
    p.add_argument("--tag", default="early")
    p.add_argument("--max_iters", default=3, type=int)
    p.add_argument("--out", default=os.path.join("exp_results", "order_repo_faces"))
    args = p.parse_args()
    CKPTS.update(dict(s.split("=", 1) for s in args.ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    A, B = "Male", "Smiling"
    fA, _ = load_ign(os.path.join(args.runs_dir, CKPTS[A]), device)
    fB, _ = load_ign(os.path.join(args.runs_dir, CKPTS[B]), device)
    base, _ = load_ign(os.path.join(args.runs_dir, BASELINES[(A, B)]), device)
    set_inference_mode([fA, fB, base], eval_mode=False)

    x0 = tiles_from_grid(args.png).to(device)
    half_ab, ab = run_order(fA, fB, x0, args.max_iters, len(x0))
    half_ba, ba = run_order(fB, fA, x0, args.max_iters, len(x0))
    samples = {"x0": x0.cpu(), "half_ab": half_ab.cpu(), "half_ba": half_ba.cpu(),
               "ab": [x.cpu() for x in ab], "ba": [x.cpu() for x in ba],
               "baseline": apply(base, x0, len(x0)).cpu(), "names": (A, B), "init": "real",
               "init_desc": f"{args.desc}  [{os.path.basename(CKPTS[A])} / {os.path.basename(CKPTS[B])}]"}
    for k in range(args.max_iters):
        l2 = (ab[k] - ba[k]).flatten(1).norm(dim=1).mean().item()
        print(f"k={k + 1}  mean L2 between orders = {l2:.2f}")
    path = os.path.join(args.out, f"repo_faces_{args.tag}_grid.png")
    grid(samples, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
