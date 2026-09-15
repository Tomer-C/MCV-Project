"""Compatible-triple test: Male + Smiling + Black_Hair, relaxed steps.

The proposal's scaling phase failed earlier with an adversarial member
(Blond). This retries K=3 with the compatible set: full f_Male pass, then
relaxed Smiling and Black steps (alpha_s = 0.3, alpha_b in {0.3, 0.5}),
one and two cycles, plus threshold selection. Reports triple-joint / FID
(vs real Male&Smiling&Black test images) / yields, with random grids.
Outputs exp_results/adaptive_b16/triple/.
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

OUT = os.path.join("exp_results", "adaptive_b16", "triple")
N = 2048
N_EVAL = 512
SPEC = "Male,Smiling,Black_Hair"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    f_m, _ = load_ign("runs/Male_500_withsamples/ckpt_0075.pt", device)
    f_s, _ = load_ign("runs/Smiling_paper500/ckpt_0025.pt", device)
    f_b, _ = load_ign("runs/Black_Hair_200/ckpt_0150.pt", device)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode([f_m, f_s, f_b], eval_mode=False)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    ns = types.SimpleNamespace(data_root="./data", image_size=64)

    idx = [ATTR_TO_IDX[a] for a in ("Male", "Smiling", "Black_Hair")]

    @torch.no_grad()
    def apply(model, x, bs=256):
        return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])

    @torch.no_grad()
    def probs3(imgs, bs=256):
        ps = []
        for i in range(0, len(imgs), bs):
            ps.append(torch.sigmoid(clf(imgs[i:i + bs].to(device))).cpu())
        p = torch.cat(ps)
        return [p[:, i] for i in idx]

    ds_int = get_filtered_dataset(ns, SPEC, split="train")
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(len(ds_int), generator=g)[:min(256, len(ds_int))]
    stats_imgs = torch.stack([ds_int[i][0] for i in sel.tolist()]).to(device)
    z = sample_freq_noise(freq_stats_from_images(stats_imgs), N)

    ds_test = get_filtered_dataset(ns, SPEC, split="test")
    n_ref = min(N_EVAL, len(ds_test))
    real = collect_real_images(ds_test, n_ref, seed=0)
    f_real = extractor(real)

    def evaluate(imgs, name):
        pm, ps, pb = probs3(imgs[:N_EVAL])
        r = {"joint3": float(((pm > 0.5) & (ps > 0.5) & (pb > 0.5)).float().mean()),
             "per": {"Male": float(pm.mean()), "Smiling": float(ps.mean()),
                     "Black_Hair": float(pb.mean())},
             "fid": frechet_distance(f_real, extractor(imgs[:N_EVAL].cpu())),
             "n_ref": n_ref}
        print(name, {k: v for k, v in r.items() if k != "per"}, flush=True)
        return r

    def relaxed_cycle(x, a_s, a_b):
        with torch.no_grad():
            x = apply(f_m, x)
            x = x + a_s * (apply(f_s, x) - x)
            x = x + a_b * (apply(f_b, x) - x)
        return x

    results = {}
    variants = {}
    for a_b in (0.3, 0.5):
        x1 = relaxed_cycle(z, 0.3, a_b)
        variants[f"t1_ab{a_b}"] = x1
        variants[f"t2_ab{a_b}"] = relaxed_cycle(x1, 0.3, a_b)
    for name, imgs in variants.items():
        results[name] = evaluate(imgs, name)
        gr = torch.Generator().manual_seed(2)
        save_image_grid(imgs[torch.randperm(len(imgs), generator=gr)[:8]].cpu(),
                        os.path.join(OUT, f"triple_{name}.png"), nrow=8)

    best = max(variants, key=lambda k: results[k]["joint3"])
    pm, ps, pb = probs3(variants[best])
    ok = ((pm > 0.5) & (ps > 0.5) & (pb > 0.5)).nonzero(as_tuple=True)[0]
    if len(ok) >= 64:
        gsel = torch.Generator().manual_seed(1)
        keep = ok[torch.randperm(len(ok), generator=gsel)][:N_EVAL]
        results[f"sel_{best}"] = evaluate(variants[best][keep], f"sel_{best}")
        results[f"sel_{best}"]["yield"] = float(len(ok)) / N
        save_image_grid(variants[best][keep[:16]].cpu(),
                        os.path.join(OUT, f"triple_sel_{best}_random16.png"), nrow=8)
    else:
        results["sel_note"] = f"only {len(ok)} qualifying -- selection skipped"

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
