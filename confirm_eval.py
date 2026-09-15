"""Fresh-seed confirmation of the winning configuration (anti winner's-curse).

The relaxed-alternation winner (f_Male@75, relaxed f_Smiling@25 step with
alpha=0.3, one cycle) was chosen as the best of several variants evaluated on
one noise batch and one real reference subset -- so its numbers may ride the
selection noise. This re-runs the FROZEN config on fresh seeds (new noise,
new FFT-stats batch, new real reference subset, new selection draw) and
reports joint / FID / diversity / yield per seed, plus the trained baseline
under the same fresh conditions for reference.

Outputs exp_results/adaptive_b11/confirm/results.json (+ random grids).
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

OUT = os.path.join("exp_results", "adaptive_b11", "confirm")
N = 2048
N_EVAL = 1024
ALPHA = 0.3
SEEDS = (1, 2, 3)


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
    ds_test = get_filtered_dataset(ns, "Male,Smiling", split="test")
    extractor = FeatureExtractor(device, fallback_classifier=clf)
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

    results = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(len(ds_int), generator=g)[:256]
        stats_imgs = torch.stack([ds_int[i][0] for i in sel.tolist()]).to(device)
        z = sample_freq_noise(freq_stats_from_images(stats_imgs), N)

        real = collect_real_images(ds_test, N_EVAL, seed=seed)
        f_real = extractor(real)
        div_real = pairwise_diversity(f_real, seed=seed)

        def evaluate(imgs, tag):
            pm, ps = probs(imgs[:N_EVAL])
            feats = extractor(imgs[:N_EVAL].cpu())
            r = {"joint": float(((pm > 0.5) & (ps > 0.5)).float().mean()),
                 "fid": frechet_distance(f_real, feats),
                 "diversity_rel_real": pairwise_diversity(feats, seed=seed) / div_real}
            print(f"seed{seed} {tag}", r, flush=True)
            return r

        with torch.no_grad():
            m1 = apply(f_m, z)
            r1 = m1 + ALPHA * (apply(f_s, m1) - m1)
            base_out = apply(baseline, z[:N_EVAL])

        entry = {"relaxed_a03": evaluate(r1, "relaxed_a03"),
                 "baseline": evaluate(base_out, "baseline")}

        pm, ps = probs(r1)
        ok = ((pm > 0.5) & (ps > 0.5)).nonzero(as_tuple=True)[0]
        gsel = torch.Generator().manual_seed(seed + 100)
        keep = ok[torch.randperm(len(ok), generator=gsel)][:N_EVAL]
        entry["sel_relaxed_a03"] = evaluate(r1[keep], "sel_relaxed_a03")
        entry["sel_relaxed_a03"]["yield"] = float(len(ok)) / len(r1)
        results[f"seed{seed}"] = entry

        gr = torch.Generator().manual_seed(seed + 200)
        save_image_grid(r1[torch.randperm(len(r1), generator=gr)[:8]].cpu(),
                        os.path.join(OUT, f"relaxed_seed{seed}_random8.png"), nrow=8)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
