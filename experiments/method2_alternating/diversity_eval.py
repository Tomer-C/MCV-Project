"""Is the classifier-selected set mode-collapsed, or only its top scorers?

Rebuilds the league pipelines (same seed) and, for each set --
real intersection images, baseline(z), m1 = f_Male(z), threshold-selected
m1 / c1 (keep ALL candidates with both P>0.5, no ranking) -- reports:

  diversity   mean pairwise L2 distance between Inception features
              (relative to the real set's value; 1.0 = as diverse as real)
  joint, fid  as before, for the threshold-selected sets

and renders, for the selected sets, a grid of 32 RANDOM members next to the
32 TOP-scored members (top rows), so the eye can compare the honest sample
with the classifier's favorites. Outputs exp_results/adaptive_b9/diversity/.
"""

import json
import os
import types

import torch

from ign.alternating import set_inference_mode
from ign.data import ATTR_TO_IDX, get_filtered_dataset
from ign.models import load_classifier, load_ign
from ign.utils import freq_stats_from_images, sample_freq_noise, save_image_grid
from ign.metrics import FeatureExtractor, collect_real_images, frechet_distance

OUT = os.path.join("exp_results", "adaptive_b9", "diversity")
N = 2048
N_EVAL = 1024


@torch.no_grad()
def pairwise_diversity(feats, n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.as_tensor(feats)[torch.randperm(len(feats), generator=g)[:n]]
    d = torch.cdist(f, f)
    return float(d.sum() / (len(f) * (len(f) - 1)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    f_m, _ = load_ign("runs/Male_500_withsamples/ckpt_0075.pt", device)
    f_s, _ = load_ign("runs/Smiling_paper500/ckpt_0025.pt", device)
    baseline, _ = load_ign("runs/Baseline_Male_AND_Smiling/ckpt_last.pt", device)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode([f_m, f_s, baseline], eval_mode=False)

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
        base_out = apply(baseline, z[:N_EVAL])

    def threshold_select(imgs, tag):
        pm, ps = probs(imgs)
        ok = ((pm > 0.5) & (ps > 0.5)).nonzero(as_tuple=True)[0]
        score = torch.minimum(pm, ps)
        gsel = torch.Generator().manual_seed(1)
        rand = ok[torch.randperm(len(ok), generator=gsel)]
        top = ok[torch.argsort(score[ok], descending=True)]
        # grid: 4 rows of TOP members, then 4 rows of RANDOM members
        grid = torch.cat([imgs[top[:32]], imgs[rand[:32]]]).cpu()
        save_image_grid(grid, os.path.join(OUT, f"top32_then_random32_{tag}.png"), nrow=8)
        keep = rand[:N_EVAL] if len(rand) >= N_EVAL else rand
        return imgs[keep], float(len(ok)) / len(imgs)

    sel_m1, y1 = threshold_select(m1, "sel_m1")
    sel_c1, y2 = threshold_select(c1, "sel_c1")

    ds_test = get_filtered_dataset(ns, "Male,Smiling", split="test")
    real = collect_real_images(ds_test, N_EVAL, seed=0)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    f_real = extractor(real)
    div_real = pairwise_diversity(f_real)

    results = {"yields": {"sel_m1": y1, "sel_c1": y2},
               "diversity_real_abs": div_real}
    sets = {"real": (real, f_real), "baseline": (base_out, None),
            "m1": (m1[:N_EVAL], None), "sel_m1": (sel_m1, None),
            "sel_c1": (sel_c1, None)}
    for name, (imgs, feats) in sets.items():
        feats = f_real if name == "real" else extractor(imgs.cpu() if imgs.is_cuda else imgs)
        entry = {"diversity_rel_real": pairwise_diversity(feats) / div_real}
        if name != "real":
            pm, ps = probs(imgs)
            entry["joint"] = float(((pm > 0.5) & (ps > 0.5)).float().mean())
            entry["fid"] = frechet_distance(f_real, feats)
        results[name] = entry
        print(name, entry, flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
