"""Diagnostic: did eval-mode BatchNorm cause the bad-looking sample grids?

For each checkpoint, renders the same 6-row grid train_ign.py saves
(z, f(z), f^2(z), f^3(z), real x, f(x)) TWICE:

  <run>_train.png  train-mode BatchNorm (batch statistics) -- the regime the
                   weights were trained in, and the only mode upstream IGN
                   ever renders with;
  <run>_eval.png   eval-mode running stats -- how the old grids were made,
                   with statistics polluted by the noise/generated batches.

It also prints the reconstruction error |f(x) - x| per mode, which is the
single number that decides the question:

  train L1 much lower than eval L1, train grid looks fine  -> the model is
      healthy; the old grids were a rendering artifact.
  both L1 high, both grids collapsed                       -> the run is
      genuinely corrupted; normalization was not the problem.

Works with the pre-paper-faithful codebase: put this file at the project root
(next to ign_models.py) and run

    python bn_check.py                              # every runs/*/ckpt_last.pt
    python bn_check.py --ckpt runs/Male/ckpt_last.pt

Checkpoints are only read; nothing is modified.
"""

import argparse
import glob
import os
import types

import torch
import torch.nn as nn

from ign_models import load_ign
from ign_utils import save_image_grid, set_seed


def set_bn_mode(model, mode):
    """train = batch statistics (momentum zeroed so buffers stay untouched)."""
    if mode == "eval":
        model.eval()
    else:
        model.train()
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = 0.0


@torch.no_grad()
def render(model, z, x, path):
    """Forward full batches (so batch statistics are decent), display first 8."""
    rows = [z[:8].clamp(-1, 1)]
    cur = z
    for _ in range(3):
        cur = model(cur)
        rows.append(cur[:8])
    recon_l1 = None
    if x is not None:
        fx = model(x)
        recon_l1 = float((fx - x).abs().mean())
        rows.append(x[:8])
        rows.append(fx[:8])
    save_image_grid(torch.cat(rows, 0), path, nrow=8)
    return recon_l1


def real_batch(ckpt, args, device):
    """A batch of test images matching the attribute the IGN was trained on."""
    try:
        from ign_data import get_filtered_dataset
        ns = types.SimpleNamespace(data_root=args.data_root, image_size=64)
        ds = get_filtered_dataset(ns, ckpt.get("attr_spec", "all"), split="test")
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(ds), generator=g)[:args.batch]
        return torch.stack([ds[i][0] for i in idx.tolist()]).to(device)
    except Exception as e:
        print(f"  (could not load real images: {e} -- noise rows only)")
        return None


def main():
    ap = argparse.ArgumentParser("eval-vs-train BatchNorm check on saved IGNs")
    ap.add_argument("--ckpt", nargs="*", default=None,
                    help="checkpoints to test; default: the Smiling run only")
    ap.add_argument("--batch", type=int, default=64,
                    help="batch size the statistics are computed from")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="bn_check_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = args.ckpt or sorted(glob.glob(os.path.join("runs", "Smiling", "ckpt_last.pt")))
    if not ckpts:
        raise SystemExit("no checkpoint found at runs/Smiling/ckpt_last.pt "
                         "(pass --ckpt explicitly for other runs)")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in ckpts:
        run = os.path.basename(os.path.dirname(path))
        print(f"== {run}: {path}")
        model, ckpt = load_ign(path, device)

        set_seed(args.seed)
        # white noise on purpose: it is what these checkpoints were trained with
        z = torch.randn(args.batch, 3, 64, 64, device=device)
        x = real_batch(ckpt, args, device)

        for mode in ("train", "eval"):
            set_bn_mode(model, mode)
            out = os.path.join(args.out_dir, f"{run}_{mode}.png")
            recon = render(model, z, x, out)
            msg = f"  {mode:5s} -> {out}"
            if recon is not None:
                msg += f"   recon |f(x)-x| = {recon:.4f}"
            print(msg)

    print("\nCompare <run>_train.png vs <run>_eval.png. If train looks fine and its"
          "\nrecon L1 is clearly lower, the old grids were a rendering artifact;"
          "\nif both are equally bad, the run itself is corrupted.")


if __name__ == "__main__":
    main()
