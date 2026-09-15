"""K=4 scaling: Male + Smiling + Black_Hair + Young, relaxed steps.

One relaxed cycle (full Male; Smiling 0.3; Black 0.5; Young 0.3) from
intersection-stats noise, quad-joint rate + FID vs real 4-way intersection,
threshold selection. Outputs exp_results/adaptive_b17/quad/.
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

OUT = os.path.join("exp_results", "adaptive_b17", "quad")
N = 2048
N_EVAL = 512
SPEC = "Male,Smiling,Black_Hair,Young"
ATTRS = ("Male", "Smiling", "Black_Hair", "Young")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    nets = {}
    for name, path in (("Male", "runs/Male_500_withsamples/ckpt_0075.pt"),
                       ("Smiling", "runs/Smiling_paper500/ckpt_0025.pt"),
                       ("Black_Hair", "runs/Black_Hair_200/ckpt_0150.pt"),
                       ("Young", "runs/Young_500/ckpt_0075.pt")):
        nets[name], _ = load_ign(path, device)
    clf, _ = load_classifier("runs/classifier/classifier.pt", device)
    set_inference_mode(list(nets.values()), eval_mode=False)
    extractor = FeatureExtractor(device, fallback_classifier=clf)
    ns = types.SimpleNamespace(data_root="./data", image_size=64)
    idx = [ATTR_TO_IDX[a] for a in ATTRS]

    @torch.no_grad()
    def apply(model, x, bs=256):
        return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])

    @torch.no_grad()
    def probs4(imgs, bs=256):
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

    with torch.no_grad():
        x = apply(nets["Male"], z)
        x = x + 0.3 * (apply(nets["Smiling"], x) - x)
        x = x + 0.5 * (apply(nets["Black_Hair"], x) - x)
        x = x + 0.3 * (apply(nets["Young"], x) - x)

    pall = probs4(x)
    ok_mask = torch.ones(len(x), dtype=torch.bool)
    for p in pall:
        ok_mask &= (p > 0.5)
    results = {"quad_raw": {
        "joint4": float(ok_mask[:N_EVAL].float().mean()),
        "per": {a: float(p.mean()) for a, p in zip(ATTRS, pall)},
        "fid": frechet_distance(f_real, extractor(x[:N_EVAL].cpu())),
        "n_ref": n_ref}}
    print("quad_raw", {k: v for k, v in results["quad_raw"].items() if k != "per"}, flush=True)

    gr = torch.Generator().manual_seed(2)
    save_image_grid(x[torch.randperm(len(x), generator=gr)[:8]].cpu(),
                    os.path.join(OUT, "quad_random8.png"), nrow=8)

    ok = ok_mask.nonzero(as_tuple=True)[0]
    if len(ok) >= 32:
        gsel = torch.Generator().manual_seed(1)
        keep = ok[torch.randperm(len(ok), generator=gsel)][:N_EVAL]
        pall_k = probs4(x[keep])
        results["quad_selected"] = {
            "joint4": 1.0, "yield": float(len(ok)) / N,
            "per": {a: float(p.mean()) for a, p in zip(ATTRS, pall_k)},
            "fid": frechet_distance(f_real, extractor(x[keep][:N_EVAL].cpu()))}
        print("quad_selected", results["quad_selected"], flush=True)
        save_image_grid(x[keep[:16]].cpu(),
                        os.path.join(OUT, "quad_sel_random16.png"), nrow=8)
    else:
        results["sel_note"] = f"only {len(ok)} qualifying"

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
