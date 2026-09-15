"""Relaxed recipe as a REAL-IMAGE editor: identity-preserving projection.

Input: 512 real test faces lacking both features (non-male, non-smiling).
Variants (alpha_m, alpha_s) over the two relaxed steps
    x <- x + alpha_m (f_Male(x) - x);  x <- x + alpha_s (f_Smiling(x) - x)
including the generation recipe's full-Male step (alpha_m = 1.0).
Metrics: joint rate, identity preservation (mean L1 to the input, SSIM-free
proxy), FID vs real intersection, one and two cycles. Grids: input row plus
each variant row for the same 8 faces.
Outputs exp_results/adaptive_b17/real_edit/.
"""

import json
import os
import types

import torch

from alternating import set_inference_mode
from ign_data import ATTR_TO_IDX, get_filtered_dataset
from ign_models import load_classifier, load_ign
from ign_utils import save_image_grid
from metrics import FeatureExtractor, collect_real_images, frechet_distance

OUT = os.path.join("exp_results", "adaptive_b17", "real_edit")
N_EVAL = 512
VARIANTS = [(1.0, 0.3), (0.7, 0.3), (0.5, 0.3), (0.5, 0.5)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    f_m, _ = load_ign("runs/Male_500_withsamples/ckpt_0075.pt", device)
    f_s, _ = load_ign("runs/Smiling_paper500/ckpt_0025.pt", device)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode([f_m, f_s], eval_mode=False)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    ns = types.SimpleNamespace(data_root="./data", image_size=64)

    im, is_ = ATTR_TO_IDX["Male"], ATTR_TO_IDX["Smiling"]

    ds_lack = get_filtered_dataset(ns, "~Male,~Smiling", split="test")
    x0 = collect_real_images(ds_lack, N_EVAL, seed=0).to(device)
    ds_int = get_filtered_dataset(ns, "Male,Smiling", split="test")
    real = collect_real_images(ds_int, N_EVAL, seed=0)
    f_real = extractor(real)

    @torch.no_grad()
    def apply(model, x, bs=256):
        return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])

    @torch.no_grad()
    def probs(imgs, bs=256):
        ps = []
        for i in range(0, len(imgs), bs):
            ps.append(torch.sigmoid(clf(imgs[i:i + bs].to(device))).cpu())
        p = torch.cat(ps)
        return p[:, im], p[:, is_]

    def cycle(x, am, as_):
        with torch.no_grad():
            x = x + am * (apply(f_m, x) - x)
            x = x + as_ * (apply(f_s, x) - x)
        return x

    results = {}
    rows = [x0[:8].cpu()]
    for am, as_ in VARIANTS:
        for n_cyc in (1, 2):
            x = x0
            for _ in range(n_cyc):
                x = cycle(x, am, as_)
            name = f"am{am}_as{as_}_c{n_cyc}"
            pm, ps = probs(x)
            results[name] = {
                "joint": float(((pm > 0.5) & (ps > 0.5)).float().mean()),
                "identity_l1": float((x - x0).abs().mean()),
                "fid": frechet_distance(f_real, extractor(x.cpu())),
            }
            print(name, results[name], flush=True)
            if n_cyc == 1:
                rows.append(x[:8].cpu())

    save_image_grid(torch.cat(rows, 0), os.path.join(OUT, "real_edit_grid.png"), nrow=8)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
