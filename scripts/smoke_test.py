"""Fast end-to-end check with random tensors -- no CelebA download needed.

Verifies shapes, the loss wiring, the alternating iteration and every metric on
a handful of CPU steps. Run this after any edit, and before submitting a GPU job.

  python scripts/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alternating import (alternating_projection, damped_alternating_projection,
                         joint_fixed_point_residual, order_disagreement, run_all_orders)
from ign_data import CELEBA_ATTRS, parse_attr_spec, spec_to_slug
from ign_models import AttributeClassifier, DCGANEnhanced
from metrics import (FeatureExtractor, attr_rates, frechet_distance,
                     kernel_distance, paired_distances)

DEV = torch.device("cpu")
OK = "  ok"


def check(name):
    print(f"[{name}]")


def main():
    torch.manual_seed(0)

    check("attribute spec parsing")
    assert parse_attr_spec("Smiling") == [(CELEBA_ATTRS.index("Smiling"), 1)]
    got = parse_attr_spec("Smiling,-Male")
    assert got == [(CELEBA_ATTRS.index("Smiling"), 1), (CELEBA_ATTRS.index("Male"), 0)], got
    assert spec_to_slug("Smiling,-Male") == "Smiling_AND_not_Male"
    try:
        parse_attr_spec("NotARealAttribute")
        raise AssertionError("should have rejected an unknown attribute")
    except ValueError:
        pass
    print(OK)

    check("IGN forward / self-composition")
    f = DCGANEnhanced(norm="batch").to(DEV).eval()
    x = torch.randn(4, 3, 64, 64)
    y = f(x)
    assert y.shape == x.shape, y.shape
    assert y.min() >= -1 and y.max() <= 1, "tanh output must be in [-1,1]"
    assert f(y).shape == y.shape, "f must be composable with itself"
    _, b = f.forward_with_bottleneck(x)
    print(f"{OK} (out {tuple(y.shape)}, bottleneck {tuple(b.shape)})")

    check("training step (loss wiring + backward)")
    from ign_utils import LossManager

    class A:  # minimal args stand-in
        in_manifold_loss_factor = 1.0
        out_of_manifold_loss_factor = 1.0
        discriminative_loss_factor = 10.0
        discriminative_loss_clamp_ratio = 1.0
        discriminative_loss_clamp_val = 9999.0
        epoch_begin_idempotence_losses = 0
        lr, beta1, beta2 = 1e-4, 0.5, 0.999
        use_amp = False
        grad_clip = True
        loss_p = 1.0
        attr_spec = "Smiling"

    f_tr = DCGANEnhanced(norm="batch").to(DEV).train()
    f_copy = DCGANEnhanced(norm="batch").to(DEV).train()
    f_copy.load_state_dict(f_tr.state_dict())
    lm = LossManager(f_tr, A())

    before = [p.detach().clone() for p in f_tr.parameters()]
    for _ in range(2):
        xb = torch.randn(4, 3, 64, 64)
        zb = torch.randn_like(xb)
        f_copy.load_state_dict(f_tr.state_dict())
        fx, fz = f_tr(xb), f_tr(zb)
        f_z = fz.detach()
        lm.add("loss_in", (fx - xb).abs().pow(1.0).mean())
        lm.add("loss_out", (f_copy(fz) - fz).abs().pow(1.0).mean())
        lm.add("loss_out_d", -lm.clamp_d((f_tr(f_z) - f_z).abs().pow(1.0)).mean())
        lm.step(epoch=0)
    changed = sum(1 for p, q in zip(before, f_tr.parameters()) if not torch.equal(p, q))
    assert changed > 0, "optimiser did not update any parameters"
    assert all(torch.isfinite(torch.tensor(v)) for v in lm._last.values()), lm._last
    print(f"{OK} ({changed} tensors updated, {lm.last_str()})")

    check("alternating projection")
    fa = DCGANEnhanced(norm="batch").to(DEV).eval()
    fb = DCGANEnhanced(norm="batch").to(DEV).eval()
    x0 = torch.randn(6, 3, 64, 64)

    # A freshly initialised DCGAN collapses every input to ~0 (weights are
    # N(0,0.02) through a deep encoder), which would make the convergence
    # diagnostics look perfect for the wrong reason. Use analytic toy
    # projections instead so the metrics are exercised on real dynamics.
    class ToyProj(torch.nn.Module):
        """Contraction toward a fixed target: a genuine non-trivial projection."""
        def __init__(self, target, rate=0.5):
            super().__init__()
            self.register_buffer("target", target)
            self.rate = rate

        def forward(self, x):
            return x + self.rate * (self.target - x)

    ta = torch.zeros(1, 3, 64, 64); ta[:, 0] = 0.8      # "red-ish" set
    tb = torch.zeros(1, 3, 64, 64); tb[:, 1] = 0.8      # "green-ish" set
    toy_a, toy_b = ToyProj(ta), ToyProj(tb)
    toy = alternating_projection([toy_a, toy_b], x0, num_iters=12)
    steps = toy["metrics"]["step_norm"].mean(1)
    assert steps[-1] < steps[0], f"contractions must shrink the step: {steps}"
    assert steps[-1] < 1e-2, f"toy iteration should converge, ended at {steps[-1]}"
    jr_toy = joint_fixed_point_residual([toy_a, toy_b], toy["final"]).mean()
    print(f"  toy contraction: step {steps[0]:.3f} -> {steps[-1]:.2e}, "
          f"joint residual {jr_toy:.3f} (nonzero: targets differ, as expected)")

    res = alternating_projection([fa, fb], x0, num_iters=5)
    assert res["final"].shape == x0.shape
    assert len(res["trajectory"]) == 6, len(res["trajectory"])
    for key in ["step_norm", "image_norm", "cycle_distance",
                "idem_residual_0", "idem_residual_1"]:
        assert res["metrics"][key].shape == (5, 6), (key, res["metrics"][key].shape)
    jr = joint_fixed_point_residual([fa, fb], res["final"])
    assert jr.shape == (6,)
    print(f"{OK} (step {res['metrics']['step_norm'][0].mean():.3f} -> "
          f"{res['metrics']['step_norm'][-1].mean():.3f}, joint residual {jr.mean():.3f})")

    check("limit-cycle detection (the 'zigzagging' risk)")
    # Two maps whose composition is a pure 2-cycle: x -> -x -> x -> ...
    # cycle_distance must collapse to ~0 while step_norm stays large, which is
    # exactly the signature the report needs to distinguish oscillation from
    # convergence.
    class Negate(torch.nn.Module):
        def forward(self, x):
            return -x

    class Shift(torch.nn.Module):
        def forward(self, x):
            return x + 0.0

    cyc = alternating_projection([Negate(), Shift()], x0, num_iters=6)
    cyc_step = cyc["metrics"]["step_norm"].mean(1)
    cyc_dist = cyc["metrics"]["cycle_distance"].mean(1)
    assert cyc_step[-1] > 1.0, f"a 2-cycle must keep moving, got {cyc_step[-1]}"
    assert cyc_dist[-1] < 1e-5, f"a 2-cycle must revisit earlier iterates, got {cyc_dist[-1]}"
    print(f"{OK} (step stays {cyc_step[-1]:.1f}, cycle_distance -> {cyc_dist[-1]:.2e})")

    check("early stop on cycle_eps")
    stopped = alternating_projection([fa, fb], x0, num_iters=50, cycle_eps=1e9)
    assert stopped["num_iters_run"] == 1, stopped["num_iters_run"]
    print(OK)

    check("damped iteration")
    d = damped_alternating_projection([fa, fb], x0, num_iters=4, alpha=0.5)
    assert d["metrics"]["joint_residual"].shape == (4, 6)
    print(OK)

    check("order permutations")
    orders = run_all_orders([fa, fb], x0, num_iters=3, names=["f_A", "f_B"])
    assert len(orders) == 2, list(orders)
    dis = order_disagreement(orders)
    assert len(dis) == 1
    print(f"{OK} ({list(orders)}, disagreement {list(dis.values())[0]:.3f})")

    check("3-model scaling (K>2)")
    fc = DCGANEnhanced(norm="batch").to(DEV).eval()
    r3 = alternating_projection([fa, fb, fc], x0, num_iters=3)
    assert "idem_residual_2" in r3["metrics"]
    assert len(run_all_orders([fa, fb, fc], x0, num_iters=2)) == 6
    print(OK)

    check("attribute classifier + rates")
    clf = AttributeClassifier().to(DEV).eval()
    logits = clf(x0)
    assert logits.shape == (6, 40), logits.shape
    rates = attr_rates(clf, x0, "Smiling,Eyeglasses", DEV)
    assert set(rates["per_attr"]) == {"Smiling", "Eyeglasses"}, rates["per_attr"]
    assert 0.0 <= rates["joint"] <= 1.0
    neg = attr_rates(clf, x0, "Smiling,-Male", DEV)
    assert "NOT_Male" in neg["per_attr"], neg["per_attr"]
    print(f"{OK} (joint={rates['joint']:.2f}, negation handled)")

    check("paired distances")
    pd = paired_distances(x0, fa(x0).detach(), device=DEV, use_lpips=False)
    for k in ["l2_mean", "psnr_mean", "ssim_mean"]:
        assert k in pd and pd[k] == pd[k], (k, pd)
    ident = paired_distances(x0, x0, device=DEV, use_lpips=False)
    assert ident["l2_mean"] < 1e-5, ident
    assert ident["ssim_mean"] > 0.99, ident
    print(f"{OK} (identity: l2={ident['l2_mean']:.2e} ssim={ident['ssim_mean']:.4f})")

    check("FID / KID (well-conditioned: n >> d)")
    # FID needs more samples than feature dims; exercise the maths directly in
    # a regime where it is actually valid, so a wrong result cannot hide.
    ga = torch.randn(600, 32).double()
    gb = torch.randn(600, 32).double() * 1.5 + 0.7
    self_fid = frechet_distance(ga, ga)
    cross_fid = frechet_distance(ga, gb)
    assert abs(self_fid) < 1e-6, f"FID(X,X) must be ~0, got {self_fid}"
    assert cross_fid > 1.0, f"FID between shifted/scaled gaussians should be large, got {cross_fid}"
    # symmetric up to float accumulation order, not bit-for-bit
    assert abs(frechet_distance(gb, ga) - cross_fid) < 1e-8 * cross_fid, "FID must be symmetric"
    # KID uses the UNBIASED MMD^2 estimator, which legitimately goes slightly
    # negative for same-distribution samples. Check it is near zero and far
    # below the cross-distribution value, not that it is non-negative.
    kid_self = kernel_distance(ga, ga, num_subsets=20, subset_size=100)
    kid_cross = kernel_distance(ga, gb, num_subsets=20, subset_size=100)
    assert abs(kid_self) < 0.1, f"KID(X,X) should be ~0, got {kid_self}"
    assert kid_cross > 20 * abs(kid_self), (kid_self, kid_cross)
    print(f"{OK} (FID(X,X)={self_fid:.2e}, FID(X,Y)={cross_fid:.2f}, "
          f"KID(X,X)={kid_self:.2e}, KID(X,Y)={kid_cross:.4f})")

    check("feature extractor plumbing")
    ex = FeatureExtractor(DEV, fallback_classifier=clf)
    feats = ex(torch.randn(6, 3, 64, 64))
    assert feats.shape[0] == 6 and feats.dtype == torch.float64, (feats.shape, feats.dtype)
    print(f"{OK} ({ex.kind} features, dim {feats.shape[1]})")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
