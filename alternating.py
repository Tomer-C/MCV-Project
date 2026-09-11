"""Alternating projection over trained IGNs -- the core of the proposal.

Von Neumann's theorem says that for closed subspaces A, B with orthogonal
projections P_A, P_B, the sequence (P_B P_A)^n x_0 converges to the projection
of x_0 onto A and B. IGNs are non-linear, oblique, and only approximately
idempotent, so nothing guarantees convergence here. This module runs the
iteration and instruments it heavily enough to say what actually happens:
convergence, a limit cycle, or divergence.

Diagnostics tracked per iteration
---------------------------------
  step_norm        ||x_{n+1} - x_n||        -> Cauchy behaviour / convergence rate
  idem_residual_A  ||f_A(x_n) - x_n||       -> how close x_n is to A's fixed-point set
  idem_residual_B  ||f_B(x_n) - x_n||       -> ...and to B's
  cycle_distance   min_{m<n} ||x_n - x_m||  -> detects limit cycles (zigzagging)
  image_norm       ||x_n||                  -> detects numerical explosion

A point in the intersection is a *joint* fixed point: both residuals near zero
at once. That is the operational definition this project tests.
"""

import itertools

import torch
import torch.nn as nn


def set_inference_mode(models, eval_mode):
    """Choose how BatchNorm behaves during the iteration.

    eval_mode=False (paper-style, default): train-mode batch statistics, which
    is the only mode upstream IGN ever runs in -- its figures are all rendered
    this way. BN momentum is zeroed so the loaded checkpoints' running-stat
    buffers are not mutated by the experiment.

    eval_mode=True: running statistics. Deterministic per-image, but an IGN's
    running stats are a blend of real, noise and generated batches, so outputs
    can be badly degraded.
    """
    for f in models:
        if eval_mode:
            f.eval()
        else:
            f.train()
            for m in f.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.momentum = 0.0


@torch.no_grad()
def apply_cycle(models, x):
    """One full sweep: x -> f_k(...f_2(f_1(x))). `models` is ordered."""
    for f in models:
        x = f(x)
    return x


@torch.no_grad()
def idempotency_residual(f, x, reduction="per_image"):
    """||f(x) - x||, the distance from x to f's fixed-point set."""
    r = (f(x) - x).abs()
    if reduction == "per_image":
        return r.flatten(1).mean(1)
    return r.mean()


@torch.no_grad()
def alternating_projection(models, x0, num_iters=20, track_trajectory=True,
                           trajectory_every=1, cycle_eps=None, eval_mode=False):
    """Run x_{n+1} = (f_K o ... o f_1)(x_n) and record diagnostics.

    Parameters
    ----------
    models : list of nn.Module mapping [-1,1]^(3,H,W) to itself.
    x0     : (B, 3, H, W) starting batch -- noise, or real test images.
    num_iters : number of full alternating sweeps.
    track_trajectory : keep every `trajectory_every`-th iterate for figures.
    eval_mode : BatchNorm behaviour, see `set_inference_mode`.

    Returns a dict with 'final', 'trajectory' (list of tensors) and 'metrics'
    (dict of per-iteration tensors, each (num_iters, B) or (num_iters,)).
    """
    set_inference_mode(models, eval_mode)

    x = x0.clone()
    trajectory = [x.clone()] if track_trajectory else []
    history = [x.clone()]  # for cycle detection; kept on the compute device

    metrics = {k: [] for k in
               ["step_norm", "image_norm", "cycle_distance"]}
    for i in range(len(models)):
        metrics[f"idem_residual_{i}"] = []

    for n in range(num_iters):
        x_prev = x
        x = apply_cycle(models, x_prev)

        metrics["step_norm"].append((x - x_prev).flatten(1).norm(dim=1).cpu())
        metrics["image_norm"].append(x.flatten(1).norm(dim=1).cpu())
        for i, f in enumerate(models):
            metrics[f"idem_residual_{i}"].append(idempotency_residual(f, x).cpu())

        # distance to the closest earlier iterate: small + non-decreasing
        # step_norm together indicate a limit cycle rather than a fixed point
        dists = torch.stack([(x - h).flatten(1).norm(dim=1) for h in history], 0)
        metrics["cycle_distance"].append(dists.min(0).values.cpu())
        history.append(x.clone())
        if len(history) > 32:            # bound memory on long runs
            history.pop(0)

        if track_trajectory and (n + 1) % trajectory_every == 0:
            trajectory.append(x.clone())

        if cycle_eps is not None and metrics["step_norm"][-1].max() < cycle_eps:
            break

    metrics = {k: torch.stack(v, 0) for k, v in metrics.items() if v}
    return {"final": x, "trajectory": trajectory, "metrics": metrics,
            "num_iters_run": len(metrics["step_norm"])}


@torch.no_grad()
def joint_fixed_point_residual(models, x):
    """max_i ||f_i(x) - x||: how badly x fails to be a joint fixed point.

    This is the scalar the alternating scheme is trying to drive to zero, and
    the cleanest single number for 'is x in the intersection'.
    """
    residuals = torch.stack([idempotency_residual(f, x) for f in models], 0)
    return residuals.max(0).values


@torch.no_grad()
def run_all_orders(models, x0, num_iters=20, names=None, eval_mode=False):
    """Run the iteration under every permutation of the models.

    The proposal asks whether the order of application matters. For orthogonal
    projections onto subspaces, the limit of von Neumann's iteration is the same
    regardless of order; if these IGNs behave like projections we would expect
    the permutations to agree, and any disagreement quantifies how far from
    projections they are.
    """
    if names is None:
        names = [f"f{i}" for i in range(len(models))]
    results = {}
    for perm in itertools.permutations(range(len(models))):
        key = " o ".join(names[i] for i in reversed(perm))  # f_B o f_A applies A first
        ordered = [models[i] for i in perm]
        results[key] = alternating_projection(ordered, x0, num_iters=num_iters,
                                              track_trajectory=False, eval_mode=eval_mode)
    return results


@torch.no_grad()
def order_disagreement(order_results):
    """Pairwise mean L2 distance between the limits reached by each order."""
    keys = list(order_results)
    table = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = order_results[keys[i]]["final"], order_results[keys[j]]["final"]
            d = (a - b).flatten(1).norm(dim=1).mean().item()
            table[f"{keys[i]}  vs  {keys[j]}"] = d
    return table


@torch.no_grad()
def damped_alternating_projection(models, x0, num_iters=20, alpha=0.5, eval_mode=False):
    """Relaxed iteration x_{n+1} = (1-a) x_n + a (f_K o ... o f_1)(x_n).

    Under-relaxation is the standard remedy when alternating projections
    oscillate between sets; it is the natural first thing to try if the
    undamped iteration zigzags, which the proposal lists as a risk.
    """
    set_inference_mode(models, eval_mode)
    x = x0.clone()
    metrics = {"step_norm": [], "joint_residual": []}
    for _ in range(num_iters):
        x_new = (1 - alpha) * x + alpha * apply_cycle(models, x)
        metrics["step_norm"].append((x_new - x).flatten(1).norm(dim=1).cpu())
        x = x_new
        metrics["joint_residual"].append(joint_fixed_point_residual(models, x).cpu())
    return {"final": x, "metrics": {k: torch.stack(v, 0) for k, v in metrics.items()}}
