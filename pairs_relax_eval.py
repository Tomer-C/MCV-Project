"""The relaxed-alternation recipe generalized to other feature pairs.

Recipe (frozen from the Male+Smiling winner): full pass of the ROBUST
feature's net at its projector stage, then a RELAXED step of the fragile
feature's net: x <- m1 + alpha (f_B(m1) - m1), alpha in {0.3, 0.5}; noise is
FFT-matched to the pair's intersection subset. Threshold-selection layered on
the better alpha.

Pairs (robust first / fragile relaxed):
  Smiling + Blond_Hair   (compatible pair)
  Male + Blond_Hair      (the adversarial pair -- does relaxation crack it?)
  Male + Young           (weak-partner pair)

Per variant: joint rate, FID vs the pair's real test intersection (reference
size = min(512, available) -- small for rare intersections, comparable within
a pair only), yields. Random-sample grids per variant.
Outputs exp_results/adaptive_b12/pairs_relax/.
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

OUT = os.path.join("exp_results", "adaptive_b12", "pairs_relax")
N = 2048
N_EVAL = 512
ALPHAS = (0.3, 0.5)

CKPTS = {"Male": "runs/Male_500_withsamples/ckpt_0075.pt",
         "Smiling": "runs/Smiling_paper500/ckpt_0025.pt",
         "Blond_Hair": "runs/Blond_Hair_500/ckpt_0075.pt",
         "Young": "runs/Young_500/ckpt_0075.pt"}
PAIRS = [("Smiling", "Blond_Hair"), ("Male", "Blond_Hair"), ("Male", "Young")]

# optional JSON config as argv[1]: {"ckpts": {...}, "pairs": [[a,b],...], "out": "..."}
import sys  # noqa: E402
if len(sys.argv) > 1:
    _cfg = json.load(open(sys.argv[1]))
    CKPTS.update(_cfg.get("ckpts", {}))
    PAIRS = [tuple(p) for p in _cfg.get("pairs", PAIRS)]
    OUT = _cfg.get("out", OUT)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    nets = {}
    for k, p in CKPTS.items():
        nets[k], _ = load_ign(p, device)
    set_inference_mode(list(nets.values()), eval_mode=False)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    ns = types.SimpleNamespace(data_root="./data", image_size=64)

    @torch.no_grad()
    def apply(model, x, bs=256):
        return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])

    @torch.no_grad()
    def pair_probs(imgs, ia, ib, bs=256):
        ps = []
        for i in range(0, len(imgs), bs):
            ps.append(torch.sigmoid(clf(imgs[i:i + bs].to(device))).cpu())
        p = torch.cat(ps)
        return p[:, ia], p[:, ib]

    results = {}
    for a, b in PAIRS:
        tag = f"{a}__{b}"
        spec = f"{a},{b}"
        ia, ib = ATTR_TO_IDX[a], ATTR_TO_IDX[b]
        ds_int = get_filtered_dataset(ns, spec, split="train")
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(len(ds_int), generator=g)[:min(256, len(ds_int))]
        stats_imgs = torch.stack([ds_int[i][0] for i in sel.tolist()]).to(device)
        z = sample_freq_noise(freq_stats_from_images(stats_imgs), N)

        ds_test = get_filtered_dataset(ns, spec, split="test")
        n_ref = min(N_EVAL, len(ds_test))
        real = collect_real_images(ds_test, n_ref, seed=0)
        f_real = extractor(real)

        def evaluate(imgs, name):
            pm, ps = pair_probs(imgs[:N_EVAL], ia, ib)
            r = {"joint": float(((pm > 0.5) & (ps > 0.5)).float().mean()),
                 "fid": frechet_distance(f_real, extractor(imgs[:N_EVAL].cpu())),
                 "n_ref": n_ref}
            print(tag, name, r, flush=True)
            return r

        with torch.no_grad():
            m1 = apply(nets[a], z)
        entry = {"m1_only": evaluate(m1, "m1_only")}
        best = (None, -1.0)
        for alpha in ALPHAS:
            with torch.no_grad():
                r1 = m1 + alpha * (apply(nets[b], m1) - m1)
            key = f"r1_a{alpha}"
            entry[key] = evaluate(r1, key)
            gr = torch.Generator().manual_seed(2)
            save_image_grid(r1[torch.randperm(len(r1), generator=gr)[:8]].cpu(),
                            os.path.join(OUT, f"{tag}_{key}.png"), nrow=8)
            if entry[key]["joint"] > best[1]:
                best = ((key, r1), entry[key]["joint"])
        key, r1 = best[0]
        pm, ps = pair_probs(r1, ia, ib)
        ok = ((pm > 0.5) & (ps > 0.5)).nonzero(as_tuple=True)[0]
        if len(ok) >= 64:
            gsel = torch.Generator().manual_seed(1)
            keep = ok[torch.randperm(len(ok), generator=gsel)][:N_EVAL]
            entry[f"sel_{key}"] = evaluate(r1[keep], f"sel_{key}")
            entry[f"sel_{key}"]["yield"] = float(len(ok)) / len(r1)
            save_image_grid(r1[keep[:16]].cpu(),
                            os.path.join(OUT, f"{tag}_sel_{key}_random16.png"), nrow=8)
        else:
            entry["sel_note"] = f"only {len(ok)} qualifying candidates -- selection skipped"
            print(tag, entry["sel_note"], flush=True)
        results[tag] = entry

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
