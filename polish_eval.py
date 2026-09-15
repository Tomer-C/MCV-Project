"""Champion alternation + one final 'sharpener' pass: joint rate vs FID.

The champion configuration (f_Male@e75 o f_Smiling@e25, 4 cycles, x0 = FFT
noise with intersection statistics) wins on joint attributes (0.60) but pays
~3x FID vs a single f_Male pass. In-loop sharpener schedules make blur worse;
this evaluates the post-hoc alternative: after the 4 cycles, apply ONE pass of
a late (sharp autoencoder) checkpoint and measure what it does to both
metrics.

Variants: no polish | +f_Male@499 | +f_Smiling@499 | +f_Male@499 then
f_Smiling@499. Outputs (json + comparison grid) to
exp_results/adaptive_b6/polish/. Run from the project directory on the VM.
"""

import json
import os
import types

import torch

from alternating import set_inference_mode
from ign_data import ATTR_TO_IDX, get_filtered_dataset
from ign_models import load_classifier, load_ign
from ign_utils import freq_stats_from_images, sample_freq_noise, save_image_grid
from metrics import FeatureExtractor, collect_real_images, frechet_distance

OUT = os.path.join("exp_results", "adaptive_b6", "polish")
N = 1024


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    fa, _ = load_ign("runs/Male_500_withsamples/ckpt_0075.pt", device)
    fb, _ = load_ign("runs/Smiling_paper500/ckpt_0025.pt", device)
    pol_m, _ = load_ign("runs/Male_500_withsamples/ckpt_0499.pt", device)
    pol_s, _ = load_ign("runs/Smiling_paper500/ckpt_0499.pt", device)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode([fa, fb, pol_m, pol_s], eval_mode=False)

    ns = types.SimpleNamespace(data_root="./data", image_size=64)
    ds_int = get_filtered_dataset(ns, "Male,Smiling", split="train")
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(ds_int), generator=g)[:256]
    stats_imgs = torch.stack([ds_int[i][0] for i in sel.tolist()]).to(device)
    x0 = sample_freq_noise(freq_stats_from_images(stats_imgs), N)

    im, is_ = ATTR_TO_IDX["Male"], ATTR_TO_IDX["Smiling"]

    @torch.no_grad()
    def joint(imgs, bs=256):
        ps = []
        for i in range(0, len(imgs), bs):
            ps.append(torch.sigmoid(clf(imgs[i:i + bs].to(device))).cpu())
        p = torch.cat(ps)
        return float(((p[:, im] > 0.5) & (p[:, is_] > 0.5)).float().mean())

    @torch.no_grad()
    def apply(model, x, bs=256):
        return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])

    with torch.no_grad():
        cur = x0
        for _ in range(4):
            cur = apply(fb, apply(fa, cur))
    variants = {
        "champion_no_polish": cur,
        "polish_Male499": apply(pol_m, cur),
        "polish_Smiling499": apply(pol_s, cur),
        "polish_M499_then_S499": apply(pol_s, apply(pol_m, cur)),
    }

    ds_test = get_filtered_dataset(ns, "Male,Smiling", split="test")
    real = collect_real_images(ds_test, min(N, len(ds_test)), seed=0)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    f_real = extractor(real)

    results = {"real_intersection_joint": joint(real)}
    for name, imgs in variants.items():
        results[name] = {"joint": joint(imgs),
                         "fid": frechet_distance(f_real, extractor(imgs.cpu()))}
        print(name, results[name], flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    rows = [x0[:8].clamp(-1, 1)] + [v[:8] for v in variants.values()]
    save_image_grid(torch.cat([r.cpu() for r in rows], 0),
                    os.path.join(OUT, "polish_grid.png"), nrow=8)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
