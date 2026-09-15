"""Pushing the SELECTION-FREE relaxed recipe: every image kept, no filtering.

Base: m1 = f_Male@75(z), z = intersection-stats FFT noise. Variants:

  fine alpha sweep    x = m1 + a (f_S(m1) - m1),  a in {0.2 .. 0.6}
  multi small steps   repeated smile-only refinements WITHOUT re-running
                      f_Male (the Male re-pass is what costs blur):
                      2 x 0.2, 3 x 0.15 (f_S recomputed at each new x)
  adaptive stepping   per-image small steps (a=0.2, max 4): an image stops
                      as soon as the classifier scores its smile > 0.5.
                      NOT selection -- all 1024 images kept (yield 1.0);
                      the classifier steers per-image effort only.

Metrics: joint, mean P(smile), FID, and for adaptive the mean steps used.
Outputs exp_results/adaptive_b19/nosel/ (results.json + random grids).
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

OUT = os.path.join("exp_results", "adaptive_b19", "nosel")
N = 1024


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

    ds_int = get_filtered_dataset(ns, "Male,Smiling", split="train")
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(ds_int), generator=g)[:256]
    stats_imgs = torch.stack([ds_int[i][0] for i in sel.tolist()]).to(device)
    z = sample_freq_noise(freq_stats_from_images(stats_imgs), N)

    ds_test = get_filtered_dataset(ns, "Male,Smiling", split="test")
    real = collect_real_images(ds_test, N, seed=0)
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

    def evaluate(x, name, extra=None):
        pm, ps = probs(x)
        r = {"joint": float(((pm > 0.5) & (ps > 0.5)).float().mean()),
             "p_smile": float(ps.mean()),
             "fid": frechet_distance(f_real, extractor(x.cpu()))}
        if extra:
            r.update(extra)
        results[name] = r
        print(name, r, flush=True)
        gr = torch.Generator().manual_seed(2)
        save_image_grid(x[torch.randperm(len(x), generator=gr)[:8]].cpu(),
                        os.path.join(OUT, f"nosel_{name}.png"), nrow=8)

    results = {}
    with torch.no_grad():
        m1 = apply(f_m, z)

        for a in (0.2, 0.3, 0.4, 0.5, 0.6):
            evaluate(m1 + a * (apply(f_s, m1) - m1), f"a{a}")

        x = m1.clone()
        for _ in range(2):
            x = x + 0.2 * (apply(f_s, x) - x)
        evaluate(x, "steps_2x0.2")

        x = m1.clone()
        for _ in range(3):
            x = x + 0.15 * (apply(f_s, x) - x)
        evaluate(x, "steps_3x0.15")

        # adaptive: small steps, per-image stop once the smile registers
        x = m1.clone()
        active = torch.ones(N, dtype=torch.bool, device=device)
        steps_used = torch.zeros(N)
        for _ in range(4):
            _, ps = probs(x)
            active = active & (ps <= 0.5).to(device)
            if not active.any():
                break
            delta = apply(f_s, x[active]) - x[active]
            x[active] = x[active] + 0.2 * delta
            steps_used[active.cpu()] += 1
        evaluate(x, "adaptive_0.2x4",
                 extra={"mean_steps": float(steps_used.mean()), "yield": 1.0})

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
