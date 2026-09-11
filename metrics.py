"""Evaluation metrics for comparing alternating projection against the baseline.

Three families, answering three different questions:

  1. Attribute satisfaction (attr_rates, joint_attr_rate)
     "Does x* actually have both features?" -- the primary claim. Uses the
     independently trained classifier.

  2. Distribution match (fid, kid)
     "Do the alternating outputs look like real A-and-B images?" FID is
     computed on InceptionV3 pool3 features when torchvision weights are
     available, and falls back to the attribute-classifier's penultimate
     features otherwise, so the pipeline never hard-fails offline. The
     fallback is clearly labelled in the output.

  3. Pairwise agreement (paired_distances)
     "Does the zero-shot x* match what the trained f_AnB produces from the
     same x_0?" -- the head-to-head the proposal asks for. LPIPS if the lpips
     package is installed, otherwise L2/PSNR/SSIM.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ign_data import ATTR_TO_IDX, parse_attr_spec


# --------------------------------------------------------------------------
# 1. attribute satisfaction
# --------------------------------------------------------------------------

@torch.no_grad()
def attr_probabilities(classifier, images, device, batch_size=128):
    """Sigmoid probabilities for all 40 attributes, (N, 40)."""
    classifier.eval()
    out = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        out.append(torch.sigmoid(classifier(batch)).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def attr_rates(classifier, images, spec, device):
    """Fraction of images satisfying each condition in `spec`, and all jointly.

    Returns {'per_attr': {name: rate}, 'joint': float, 'mean_prob': {...}}.
    'joint' is the headline number: the share of outputs that land in the
    intersection according to the classifier.
    """
    probs = attr_probabilities(classifier, images, device)
    conditions = parse_attr_spec(spec)
    idx_to_name = {v: k for k, v in ATTR_TO_IDX.items()}

    satisfied = torch.ones(probs.shape[0], dtype=torch.bool)
    per_attr, mean_prob = {}, {}
    for idx, want in conditions:
        pred = (probs[:, idx] > 0.5).long()
        ok = (pred == want)
        satisfied &= ok
        label = idx_to_name[idx] if want == 1 else f"NOT_{idx_to_name[idx]}"
        per_attr[label] = ok.float().mean().item()
        # probability of the *wanted* class, so higher is always better
        mean_prob[label] = (probs[:, idx] if want == 1 else 1 - probs[:, idx]).mean().item()

    return {"per_attr": per_attr,
            "joint": satisfied.float().mean().item(),
            "mean_prob": mean_prob,
            "n": int(probs.shape[0])}


# --------------------------------------------------------------------------
# 2. distribution match
# --------------------------------------------------------------------------

class FeatureExtractor:
    """InceptionV3 pool3 features, with a graceful offline fallback."""

    def __init__(self, device, fallback_classifier=None):
        self.device = device
        self.kind = None
        self.model = None
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            net.fc = torch.nn.Identity()
            net.eval().to(device)
            self.model, self.kind = net, "inception"
        except Exception as e:
            if fallback_classifier is None:
                raise RuntimeError(
                    f"Inception weights unavailable ({e}) and no fallback classifier given."
                )
            self.model, self.kind = fallback_classifier, "attr_classifier"
            print(f"[metrics] Inception unavailable ({type(e).__name__}); "
                  "using attribute-classifier features. FID values are then only "
                  "comparable within this run, not to published FIDs.")

    @torch.no_grad()
    def __call__(self, images, batch_size=64):
        feats = []
        for i in range(0, images.shape[0], batch_size):
            x = images[i:i + batch_size].to(self.device).clamp(-1, 1)
            if self.kind == "inception":
                # inception expects 299x299, ImageNet-normalised
                x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
                x = (x + 1) / 2
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std
                feats.append(self.model(x).cpu())
            else:
                feats.append(self.model.features(x).flatten(1).cpu())
        return torch.cat(feats, 0).double()


def _sqrtm_psd(mat, eps=0.0):
    """Matrix square root of a symmetric PSD matrix via eigendecomposition.

    Avoids the scipy dependency that reference FID implementations use.
    Negative eigenvalues are pure numerical noise on a PSD input, so they are
    clamped to zero rather than to a positive floor -- a positive floor would
    inflate Tr((Sa Sb)^.5) and make FID(X, X) come out negative.
    """
    mat = (mat + mat.T) / 2
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp(min=eps)
    return (vecs * vals.sqrt().unsqueeze(0)) @ vecs.T


def frechet_distance(feats_a, feats_b, warn_rank=True):
    """FID between two feature sets: ||mu_a-mu_b||^2 + Tr(Sa + Sb - 2(Sa Sb)^.5).

    Note on sample size: with fewer samples than feature dimensions the
    covariances are rank-deficient and FID is severely biased (it can even go
    slightly negative from floating-point error). InceptionV3 gives 2048-dim
    features, so use at least ~2048 samples -- and preferably 10k -- for numbers
    you intend to report. A warning is emitted when the estimate is unreliable.
    """
    n_a, n_b = feats_a.shape[0], feats_b.shape[0]
    d = feats_a.shape[1]
    if warn_rank and min(n_a, n_b) < d:
        print(f"[metrics] WARNING: FID with n={min(n_a, n_b)} samples < d={d} feature "
              "dims is rank-deficient and strongly biased. Increase --num_samples "
              "(>= 2048, ideally 10000) before reporting these numbers.")

    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    sa = torch.cov(feats_a.T)
    sb = torch.cov(feats_b.T)
    diff = mu_a - mu_b

    # sqrtm(Sa) Sb sqrtm(Sa) is PSD, so its trace-sqrt is real and non-negative
    sqrt_sa = _sqrtm_psd(sa)
    covmean = _sqrtm_psd(sqrt_sa @ sb @ sqrt_sa)
    fid = float(diff.dot(diff) + torch.trace(sa) + torch.trace(sb) - 2 * torch.trace(covmean))
    # clamp away residual float error around the FID(X,X)=0 case
    return max(fid, 0.0)


def kernel_distance(feats_a, feats_b, num_subsets=50, subset_size=None):
    """Unbiased KID with a polynomial kernel; more stable than FID at small N.

    Worth reporting alongside FID because the A-and-B subset can be small and
    FID is badly biased for a few thousand samples.

    This is the UNBIASED MMD^2 estimator, so for two samples from the same
    distribution it fluctuates around zero and may come out slightly NEGATIVE.
    That is correct behaviour, not an error: interpret |KID| near zero as
    "indistinguishable", and only compare KID values against each other.
    """
    n = min(feats_a.shape[0], feats_b.shape[0])
    subset_size = subset_size or min(1000, n)
    d = feats_a.shape[1]
    vals = []
    for _ in range(num_subsets):
        x = feats_a[torch.randperm(feats_a.shape[0])[:subset_size]]
        y = feats_b[torch.randperm(feats_b.shape[0])[:subset_size]]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        m = subset_size
        vals.append(((kxx.sum() - kxx.diag().sum()) / (m * (m - 1))
                     + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
                     - 2 * kxy.mean()).item())
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# 3. pairwise agreement
# --------------------------------------------------------------------------

@torch.no_grad()
def paired_distances(a, b, device=None, use_lpips=True):
    """Per-pair image distances between two aligned batches (same x_0)."""
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    a, b = a.clamp(-1, 1), b.clamp(-1, 1)
    out = {}

    l2 = (a - b).flatten(1).norm(dim=1)
    out["l2_mean"] = l2.mean().item()
    out["l2_std"] = l2.std().item()

    mse = (a - b).pow(2).flatten(1).mean(1)
    out["psnr_mean"] = (10 * torch.log10(4.0 / mse.clamp(min=1e-10))).mean().item()
    out["ssim_mean"] = float(_ssim(a, b).mean())

    if use_lpips:
        try:
            import lpips
            net = lpips.LPIPS(net="alex").to(device or "cpu")
            scores = []
            for i in range(0, a.shape[0], 32):
                scores.append(net(a[i:i+32].to(device or "cpu"),
                                  b[i:i+32].to(device or "cpu")).flatten().cpu())
            out["lpips_mean"] = torch.cat(scores).mean().item()
        except Exception as e:
            out["lpips_mean"] = None
            out["lpips_error"] = f"{type(e).__name__}: {e}"
    return out


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.T @ g).expand(channels, 1, size, size).contiguous()
    return window


def _ssim(a, b, size=11):
    """Standard SSIM on [-1,1] images (dynamic range 2)."""
    channels = a.shape[1]
    window = _gaussian_window(size, channels=channels, device=a.device)
    pad = size // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a2
    sb = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b2
    sab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return ssim_map.flatten(1).mean(1).cpu()


@torch.no_grad()
def collect_real_images(dataset, n, device="cpu", seed=0):
    """Sample n real images from a dataset as a single tensor, for FID references."""
    g = torch.Generator().manual_seed(seed)
    n = min(n, len(dataset))
    idx = torch.randperm(len(dataset), generator=g)[:n]
    return torch.stack([dataset[int(i)][0] for i in idx], 0)
