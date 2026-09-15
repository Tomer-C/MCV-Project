"""Zero-shot pipelines aiming for the trained baseline's league.

Motivation: f_Male@e75 alone already generates at baseline-league realism
(FID ~48 vs the trained f_{Male and Smiling} baseline's ~53); it only lacks
smiles (~0.5). This evaluates minimal-pass and selection-based zero-shot
pipelines against the baseline on BOTH metrics:

  m1              f_Male(z)                       1 pass
  c1              f_Smiling(f_Male(z))            1 cycle  (2 passes)
  c1_pol          f_Smiling@499 polish of c1      3 passes
  c2, c4          2 / 4 alternation cycles
  cim2, cim4      Cimmino-style SIMULTANEOUS projection:
                  x <- (f_Male(x) + f_Smiling(x)) / 2, 2 / 4 rounds
  sel_m1          rejection sampling: f_Male(z) candidates, keep those the
                  classifier scores >0.5 on BOTH features (black-box use of
                  the trained parts -- no intersection training anywhere)
  sel_c1          same selection applied to c1 outputs
  baseline        f_{Male and Smiling}(z), the trained reference

Per variant: joint rate, FID vs real test intersection images, selection
yield where applicable. Outputs exp_results/adaptive_b7/league/: results.json,
one 8-sample strip per variant (league_<name>.png) and a stacked comparison
grid (league_grid.png). Run from the project directory on the VM.
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

OUT = os.path.join("exp_results", "adaptive_b7", "league")
N = 2048          # candidates generated
N_EVAL = 1024     # images scored per variant (selections keep this many)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    f_m, _ = load_ign("runs/Male_500_withsamples/ckpt_0075.pt", device)
    f_s, _ = load_ign("runs/Smiling_paper500/ckpt_0025.pt", device)
    f_s_late, _ = load_ign("runs/Smiling_paper500/ckpt_0499.pt", device)
    baseline, _ = load_ign("runs/Baseline_Male_AND_Smiling/ckpt_last.pt", device)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode([f_m, f_s, f_s_late, baseline], eval_mode=False)

    ns = types.SimpleNamespace(data_root="./data", image_size=64)
    ds_int = get_filtered_dataset(ns, "Male,Smiling", split="train")
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(ds_int), generator=g)[:256]
    stats_imgs = torch.stack([ds_int[i][0] for i in sel.tolist()]).to(device)
    z = sample_freq_noise(freq_stats_from_images(stats_imgs), N)

    im, is_ = ATTR_TO_IDX["Male"], ATTR_TO_IDX["Smiling"]

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

    with torch.no_grad():
        m1 = apply(f_m, z)
        c1 = apply(f_s, m1)
        c2 = apply(f_s, apply(f_m, c1))
        c4 = apply(f_s, apply(f_m, apply(f_s, apply(f_m, c2))))
        c1_pol = apply(f_s_late, c1)
        cim = z
        cims = {}
        for k in range(1, 5):
            cim = (apply(f_m, cim) + apply(f_s, cim)) / 2
            if k in (2, 4):
                cims[k] = cim
        base_out = apply(baseline, z[:N_EVAL])

    def select(imgs):
        pm, ps = probs(imgs)
        ok = (pm > 0.5) & (ps > 0.5)
        score = torch.minimum(pm, ps)
        order = torch.argsort(score, descending=True)[:N_EVAL]
        return imgs[order], float(ok.float().mean())

    sel_m1, yield_m1 = select(m1)
    sel_c1, yield_c1 = select(c1)

    variants = {
        "m1": m1[:N_EVAL], "c1": c1[:N_EVAL], "c1_pol": c1_pol[:N_EVAL],
        "c2": c2[:N_EVAL], "c4": c4[:N_EVAL],
        "cim2": cims[2][:N_EVAL], "cim4": cims[4][:N_EVAL],
        "sel_m1": sel_m1, "sel_c1": sel_c1, "baseline": base_out,
    }

    ds_test = get_filtered_dataset(ns, "Male,Smiling", split="test")
    real = collect_real_images(ds_test, min(N_EVAL, len(ds_test)), seed=0)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    f_real = extractor(real)

    results = {"selection_yield": {"sel_m1": yield_m1, "sel_c1": yield_c1}}
    for name, imgs in variants.items():
        pm, ps = probs(imgs)
        results[name] = {
            "joint": float(((pm > 0.5) & (ps > 0.5)).float().mean()),
            "p_male": float(pm.mean()), "p_smile": float(ps.mean()),
            "fid": frechet_distance(f_real, extractor(imgs.cpu())),
        }
        print(name, results[name], flush=True)
        save_image_grid(imgs[:8].cpu(), os.path.join(OUT, f"league_{name}.png"), nrow=8)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    rows = [variants[n][:8].cpu() for n in variants]
    save_image_grid(torch.cat(rows, 0), os.path.join(OUT, "league_grid.png"), nrow=8)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
