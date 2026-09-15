"""Relaxed alternating projection: partial steps to fight composition blur.

Classical relaxed projections take x <- x + alpha (P(x) - x). Here the Male
step stays full (its output is sharp) and the Smiling step is relaxed:

  one relaxed cycle:  x1 = m1 + alpha (f_Smiling(m1) - m1),  m1 = f_Male(z)
  two relaxed cycles: x2 = x1', with x1' = f_Male(x1) relaxed-smiled again

Sweeps alpha in {0.3, 0.5, 0.7, 1.0} (1.0 = the ordinary alternation) for one
and two cycles, plus classifier threshold-selection on the best-FID variants.
Reports joint / FID / diversity per variant; grids of random samples.
Outputs exp_results/adaptive_b10/relax/. Run from the VM project directory.
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

OUT = os.path.join("exp_results", "adaptive_b10", "relax")
N = 2048
N_EVAL = 1024
ALPHAS = (0.3, 0.5, 0.7, 1.0)


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
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode([f_m, f_s], eval_mode=False)

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
        variants = {}
        for a in ALPHAS:
            x1 = m1 + a * (apply(f_s, m1) - m1)
            variants[f"r1_a{a}"] = x1
            x2m = apply(f_m, x1)
            x2 = x2m + a * (apply(f_s, x2m) - x2m)
            variants[f"r2_a{a}"] = x2

    ds_test = get_filtered_dataset(ns, "Male,Smiling", split="test")
    real = collect_real_images(ds_test, N_EVAL, seed=0)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    f_real = extractor(real)
    div_real = pairwise_diversity(f_real)

    def evaluate(imgs, name):
        pm, ps = probs(imgs)
        feats = extractor(imgs.cpu()[:N_EVAL])
        r = {"joint": float(((pm[:N_EVAL] > 0.5) & (ps[:N_EVAL] > 0.5)).float().mean()),
             "p_smile": float(ps.mean()),
             "fid": frechet_distance(f_real, feats),
             "diversity_rel_real": pairwise_diversity(feats) / div_real}
        print(name, r, flush=True)
        return r

    results = {}
    for name, imgs in variants.items():
        results[name] = evaluate(imgs[:N_EVAL], name)
        gr = torch.Generator().manual_seed(2)
        save_image_grid(imgs[torch.randperm(len(imgs), generator=gr)[:8]].cpu(),
                        os.path.join(OUT, f"relax_{name}.png"), nrow=8)

    # threshold-selection on the two best-FID relaxed variants (alpha < 1)
    by_fid = sorted((v["fid"], k) for k, v in results.items() if not k.endswith("a1.0"))
    for _, name in by_fid[:2]:
        imgs = variants[name]
        pm, ps = probs(imgs)
        ok = ((pm > 0.5) & (ps > 0.5)).nonzero(as_tuple=True)[0]
        gsel = torch.Generator().manual_seed(1)
        keep = ok[torch.randperm(len(ok), generator=gsel)][:N_EVAL]
        r = evaluate(imgs[keep], f"sel_{name}")
        r["yield"] = float(len(ok)) / len(imgs)
        results[f"sel_{name}"] = r
        save_image_grid(imgs[keep[:32]].cpu(),
                        os.path.join(OUT, f"relax_sel_{name}_random32.png"), nrow=8)

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
