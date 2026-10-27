# Method 3: commutative IGNs

Fine-tuning f_A and f_B jointly with an extra loss that pushes them to commute
(proposal §3.3, Theorem 1). Part 1 is the first experiment; Part 2 is the
follow-up hyperparameter sweep, which also corrects one conclusion of Part 1
(see [Correction to Part 1](#correction-to-part-1)).

## Part 1: first experiment

*Commuting IGNs: does training f_A and f_B to commute help alternating projection?*

Companion to [README.md](../README.md) and [RESULTS.md](RESULTS.md). This documents a
side experiment, not a replacement for the existing pipeline: **the result is
negative** on the metric that matters (joint attribute satisfaction), despite a
real, measurable effect on the metric it was designed to move (order-sensitivity).
Recorded here so the negative result doesn't have to be rediscovered.

### Motivation

`ign/alternating.py` already tests whether two independently-trained IGNs converge
into the intersection of their target sets. Von Neumann's alternating-projection
theorem gives a convergence guarantee for orthogonal projections onto convex
sets, but IGNs are non-linear, oblique, and only approximately idempotent — the
README is explicit that the theorem is "inspiration only".

There is a more specific, checkable version of that guarantee:

> **If `f`, `g` are idempotent and commute** (`f∘g = g∘f`), **then `f∘g` is
> idempotent, and `Fix(f∘g) = Fix(f) ∩ Fix(g)`.**

`f_A` and `f_B` here are each only approximately idempotent (that's what IGN
training optimizes for) and are never trained to commute with each other —
they're trained completely independently, one per attribute. The question this
experiment asks: if we explicitly add a loss term that pushes `f_A` and `f_B`
toward commuting, does alternating projection get measurably closer to the
theorem's guarantee, and does that translate into better joint-attribute
outputs?

### Method: `train/train_commuting_pair.py`

New script, warm-starts from two already-trained single-attribute checkpoints
(cheap) and fine-tunes them **jointly**, one optimizer over both nets' params,
one combined loss:

```
total = (loss_in_A + loss_out_A + loss_out_d_A)        # keeps f_A anchored to Fix(f_A) = A
      + (loss_in_B + loss_out_B + loss_out_d_B)        # keeps f_B anchored to Fix(f_B) = B
      + lambda_commute * D(f_A(f_B(x)), f_B(f_A(x)))   # pulls them toward commuting
```

The per-network IGN losses are kept deliberately — dropping them would let
`f_A = f_B` trivially satisfy the commute term, collapsing both attribute
projectors into one useless shared function. The commute term is evaluated on
real images from an **unfiltered** pool (not noise, not `A`- or `B`-only),
since `Fix(f∘g) = Fix(f)∩Fix(g)` is a whole-space statement.

```bash
python -m train.train_commuting_pair \
    --ckpt_a runs/Male_500_withsamples/ckpt_0075.pt \
    --ckpt_b runs/Blond_Hair_500/ckpt_0075.pt \
    --run_name Male_Blond_commuting_probe --num_epochs 10
```

Outputs `ckpt_A_last.pt` / `ckpt_B_last.pt`, drop-in compatible with
`experiments/method1_single_ign/run_experiment.py --models` (arch + attr_spec carried in the checkpoint, same
as `train/train_ign.py`'s output) — so every existing evaluation script works on them
unchanged.

**`infra/vm_orchestrate.py`** gained a generic `exec <script.py>` / `tail` /
`killscript` command set, for launching one-off scripts like this one on the
VM without them needing to join `infra/vm_experiments.py`'s fixed suite.

**Known limitation:** the script does not persist FFT noise statistics into
its output checkpoints (`train/train_ign.py` does; this one didn't, originally). Any
`experiments/method1_single_ign/run_experiment.py --init noise` call on a checkpoint from this script falls
back to white noise with a warning, unless `--noise_stats_spec` is also passed
to compute fresh stats from real images. All results below use either real
images or `--noise_stats_spec` for the out-of-distribution condition, not the
checkpoint fallback.

### Experimental setup

Two pairs, warm-started from each attribute's epoch-75 checkpoint (the
"projector stage" — see [TRAINING.md](TRAINING.md), projection strength peaks
early and decays with more training):

- **Male + Smiling** — a normal, non-anti-correlated pair with a real trained
  `f_{A∩B}` baseline already in `runs/Baseline_Male_AND_Smiling`, so it has a
  ground-truth comparison point.
- **Male + Blond_Hair** — the anti-correlated pair `RESULTS.md` already
  documents as failing under every existing method
  (`results/method2_alternating/limits/male_blond_tug_of_war_trajectory.png`).

10 epochs each, `lambda_commute=5.0` (untuned — a first probe, not a swept
hyperparameter), L1 for the commute distance `D`.

### Training result: the commute loss does go down

| pair | commute loss, epoch 0 → 9 | each net's own `loss_in` |
|---|---|---|
| Male + Blond_Hair | 0.141 → 0.030 | stable (~0.070 / ~0.090), no collapse |
| Male + Smiling | 0.072 → 0.024 | stable (~0.069 / ~0.068), no collapse |

![Male+Blond_Hair loss curves](../results/method3_commuting/experiment/male_blond_loss_curves.png)
![Male+Smiling loss curves](../results/method3_commuting/experiment/male_smiling_loss_curves.png)

Both networks' own attribute-projection losses stay flat and distinct
throughout — the guard against `f_A = f_B` collapse worked. Final sample
grids (rows: `x`, `f_A(x)`, `f_B(x)`, `f_A(f_B(x))`, `f_B(f_A(x))`):

![Male+Blond_Hair final samples](../results/method3_commuting/experiment/male_blond_final_samples.png)
![Male+Smiling final samples](../results/method3_commuting/experiment/male_smiling_final_samples.png)

### Evaluation result: order-sensitivity drops, but joint attribute rate doesn't improve

Order-sensitivity (`experiments/method1_single_ign/run_experiment.py`'s `order_disagreement`, pixel-distance
between the limits of `f_A∘f_B` vs `f_B∘f_A`) is the direct test of the
theorem's premise, and it *did* improve — most clearly with real-face
initialization, less so with noise. (Correction, see
[Part 2](#part-2-hyperparameter-sweep): the "→" pairs below are
the commuting-trained pair at 20 vs 1 iterations, not naive vs commuting; the
naive 1-iteration values, measured later on the same x₀, are added.)

| pair, init | order distance |
|---|---|
| Male+Blond, noise (checkpoint fallback → white) | 39.0 |
| Male+Blond, real images | 9.5 (20 iters) → 4.4 (`--num_iters 1`, the project's documented optimum); naive at 1 iter: 19.0 |
| Male+Smiling, noise (checkpoint fallback → white) | 47.2 |
| Male+Smiling, real images | 7.0 (20 iters) → 3.5 (`--num_iters 1`); naive at 1 iter: 10.9 |

But **the metric that actually matters — joint attribute satisfaction — tells
a different story**, once compared head-to-head against plain alternating
projection with no commute loss at all (both pairs' original independently-
trained checkpoints, same `f_A`/`f_B`, no fine-tuning). Full comparison for
Male+Smiling: baseline (real trained-on-intersection model), naive alternating
at 1/2/3 composition iterations, and the commuting-trained pair at 1/2/3
iterations — tested both in-distribution (real-face init) and out-of-
distribution (FFT frequency-matched noise, the paper-faithful noise type per
`TRAINING.md`, not plain white noise):

| method | in-dist. joint rate | out-of-dist. joint rate |
|---|---|---|
| baseline (`f_{A∩B}`, trained directly) | 0.59 | 0.91 |
| naive, 1 iter | 0.22 | 0.58 |
| naive, 2 iter | 0.30 | 0.46 |
| naive, 3 iter | 0.42 | 0.48 |
| commuting-trained, 1 iter | 0.22 | 0.47 |
| commuting-trained, 2 iter | 0.17 | 0.27 |
| commuting-trained, 3 iter | 0.05 | 0.12 |
| (classifier ceiling, real A∩B photos) | 0.85 | 0.85 |

**Naive alternating (no commute loss) beats the commuting-trained pair at
every iteration depth, in both domains** — and the commuting-trained pair gets
*worse* with more composition iterations while naive alternating holds
roughly flat or improves. Visually:

![Male+Smiling, in-distribution comparison](../results/method3_commuting/experiment/method_comparison_in_distribution.png)
![Male+Smiling, out-of-distribution comparison](../results/method3_commuting/experiment/method_comparison_out_of_distribution.png)

(Rows top→bottom: `x_0`, the real baseline, naive 1/2/3 iters, commuting
1/2/3 iters — same input faces across every row.)

The same naive-vs-commuting comparison for Male+Blond_Hair (no trained
baseline exists for this pair, so no baseline row):

![Male+Blond_Hair, in-distribution comparison](../results/method3_commuting/experiment/male_blond_naive_vs_commute_in_distribution.png)
![Male+Blond_Hair, out-of-distribution comparison](../results/method3_commuting/experiment/male_blond_naive_vs_commute_out_of_distribution.png)

> **Follow-up:** the λ / lr / distance-metric sweep and longer fine-tuning
> suggested below were run — see [Part 2](#part-2-hyperparameter-sweep).
> Longer fine-tuning collapses the pair onto a shared face; a 1–5 epoch,
> lr = 2e-5, λ = 5 fine-tune gives a small in-distribution gain.

### Conclusion

At this scale — 10 epochs, `lambda_commute=5` untuned, one warm-started pair
per condition — **training f_A and f_B to commute measurably reduces
order-sensitivity (the theorem's actual premise) but does not improve, and
in fact appears to hurt, joint attribute satisfaction relative to doing
nothing extra.** The likely explanation: the extra fine-tuning epochs pull
each checkpoint away from its epoch-75 "projector sweet spot"
(`TRAINING.md`), trading individual projection strength for a small gain in
mutual agreement between orders — a bad trade given the theorem's guarantee
only helps if the two functions are *also* still good, sharp projectors onto
their own target sets.

This is a real, reproducible negative result on the specific setup tested
(untuned λ, short fine-tune, epoch-75 warm start), not a proof the idea can't
work — a longer run, a lower `lambda_commute` (less disruption to each net's
own projector quality), or training from scratch as a genuinely joint
pipeline (rather than warm-started fine-tuning) are all still open. Recording
it here so the next attempt starts from this, instead of re-deriving it.

### Reproducing this

```bash
# fine-tune a pair to commute (warm-started, cheap)
python -m train.train_commuting_pair --ckpt_a <ckpt> --ckpt_b <ckpt> --run_name <name> --num_epochs 10

# evaluate: real-image init (in-distribution)
python -m experiments.method1_single_ign.run_experiment --models <ckptA> <ckptB> --names f_A f_B \
    --baseline <baseline_ckpt> --intersection_spec "<A>,<B>" \
    --init real --num_iters 1 --num_samples 256 --skip_fid --tag <name>

# evaluate: paper-faithful FFT-matched noise (out-of-distribution),
# independent of whether the checkpoint has freq_stats saved
python -m experiments.method1_single_ign.run_experiment --models <ckptA> <ckptB> --names f_A f_B \
    --baseline <baseline_ckpt> --intersection_spec "<A>,<B>" \
    --init noise --noise_stats_spec all --num_iters 1 --num_samples 256 --skip_fid --tag <name>
```

On the shared VM: `python -m infra.vm_orchestrate exec train/train_commuting_pair.py -- <args>`,
then `python -m infra.vm_orchestrate tail train/train_commuting_pair.py`.

## Part 2: hyperparameter sweep

*Commuting-loss sweep: hyperparameters and fine-tune length*

Follow-up to [Part 1](#part-1-first-experiment), which
left three things open: `lambda_commute` was untuned (5.0), fine-tuning was
short (10 epochs), and the commute distance was L1 only. This sweep covers all
three: 11 fine-tuning runs on Male+Smiling and 4 on Male+Blond_Hair, up to 50
epochs each, with a snapshot every 5 epochs (every epoch for one run), about
15 GPU-hours on one A10.

**Bottom line.**
- **Longer fine-tuning makes it worse.** Every config with a commute term
  (λ > 0) ends up mapping all inputs onto one shared face.
- **The commute term causes this, not the extra fine-tuning.** λ = 0 controls
  on the same schedule don't collapse.
- **There is one small, reproducible gain:** fine-tuning for 1–5 epochs at
  λ = 5, lr = 2e-5 raises the in-distribution joint rate by about +0.05 at
  every iteration depth, with image quality unchanged (KID). The effect
  appeared on two training seeds and was confirmed on 1024 fresh inputs.
- **No gain out of distribution, and none on the anti-correlated Male+Blond
  pair.**
- **Nothing tested gets near the trained f_A∩B baseline.**

### What changed in the code

| file | change |
|---|---|
| `train/train_commuting_pair.py` | keeps a `ckpt_{A,B}_eNNNN.pt` snapshot at every `--save_every` epochs (NNNN = epochs completed), so one long run can be evaluated at many fine-tune lengths; data loaders no longer go through `itertools.cycle`, which cached every batch it yielded in RAM |
| `experiments/method3_commuting/commuting_sweep.py` (new) | trains a JSON spec of runs one after another and evaluates each as it finishes; resumable (skips runs whose final snapshot exists) |
| `experiments/method3_commuting/commuting_sweep_eval.py` (new) | evaluates naive + every snapshot on **one fixed x₀** per domain, see below |
| `experiments/method3_commuting/make_commuting_grids.py` (new) | the image grids in the original layout, plus sweep-overview / epoch-progression grids and metric curves |
| `commuting_sweep_*.json` (new) | the sweep specs actually run |

### Evaluation

The same two domains as before: **in-distribution** (256 real CelebA test faces)
and **out-of-distribution** (256 FFT-matched noise samples, stats from train
images). Both are fixed and shared across every checkpoint. All iterates are
`(f_B∘f_A)^n(x₀)`, n = 1, 2, 3, with train-mode BatchNorm over the whole batch,
as in `experiments/method1_single_ign/run_experiment.py`. On the original checkpoints this reproduces the
earlier numbers exactly: naive 0.22 / 0.30 / 0.42, probe 0.22 / 0.17 / 0.05.

The metrics:

| metric | meaning |
|---|---|
| joint rate | classifier says Male ∧ Smiling (the headline in the earlier doc) |
| order distance | ‖(f_B f_A)(x₀) − (f_A f_B)(x₀)‖, the theorem's premise |
| sharpness | mean \|Laplacian\| relative to real faces |
| diversity | per-pixel std across the batch over the central face crop, relative to real faces; ~0 when every input lands on the same face |
| KID | Inception-feature distance to real Male ∧ Smiling test photos (final robustness pass only) |

**Why KID.** The classifier happily labels one blurry prototype face as
"Male ∧ Smiling", so a collapsing pair can *raise* the joint rate. An example
is λ = 20 at epoch 5 below: joint 0.57, but every output is the same smudge.
KID can't be gamed that way. Sharpness alone isn't enough either, because the
LPIPS variant collapses onto a *sharp* face. Diversity catches both.

### Result 1: more epochs make it strictly worse

The original config (λ = 5, lr = 1e-4, L1) extended from 10 to 50 epochs:

| epochs | in-dist joint (1 / 2 / 3 iter) | OOD joint (1 / 2 / 3) | order dist | diversity (3 iter) |
|---|---|---|---|---|
| 0 (naive) | 0.22 / 0.30 / 0.42 | 0.55 / 0.50 / 0.51 | 10.9 | 0.99 |
| 5 | 0.28 / 0.34 / 0.32 | 0.51 / 0.39 / 0.33 | 4.1 | 0.76 |
| 10 (the original probe length) | 0.27 / 0.20 / 0.09 | 0.53 / 0.32 / 0.20 | 3.7 | 0.64 |
| 25 | 0.18 / 0.00 / 0.00 | 0.45 / 0.07 / 0.00 | 2.9 | 0.45 |
| 50 | 0.01 / 0.00 / 0.00 | 0.34 / 0.00 / 0.00 | 2.1 | 0.27 |

![λ=5 over 50 epochs](../results/method3_commuting/sweep/ms_lam5_lr1e-4_progression_real_3iter.png)

Order distance keeps falling, so the commute loss is doing its job. But it gets
there by collapsing: by epoch 50 every input maps to the same blurry mean face.

### Result 2: the commute term causes it, not the fine-tuning

λ = 0 controls use the same schedule and the same joint optimizer, only
without the commute term:

| config (epoch 50) | in-dist joint (1 / 3) | OOD joint (3) | order dist | diversity |
|---|---|---|---|---|
| naive | 0.22 / 0.42 | 0.51 | 10.9 | 0.99 |
| λ = 0, lr = 1e-4 | 0.18 / 0.34 | 0.34 | 10.5 | 0.93 |
| λ = 0, lr = 2e-5 | 0.16 / 0.29 | 0.34 | 13.6 | 0.96 |
| λ = 5, lr = 1e-4 | 0.01 / 0.00 | 0.00 | 2.1 | 0.27 |

Plain fine-tuning drifts slowly downhill, the expected drift away from the
epoch-75 "projector sweet spot" (`TRAINING.md`). It stays sharp and varied,
and order distance doesn't move. Only the commute term collapses the outputs.

### Result 3: how it fails — both compositions agree on one face

The training-time sample grid for λ = 5 at epoch 49 has rows x, f_A(x),
f_B(x), f_A(f_B(x)) and f_B(f_A(x)):

![train samples, λ=5 epoch 49](../results/method3_commuting/sweep/ms_lam5_lr1e-4_train_samples_e049.png)

Each net alone still works on real faces (rows 2–3 are sharp, and each net's
`loss_in` even improves slightly, 0.071 → 0.064). What collapses is each net's output **when
it is fed the other net's output** (rows 4–5).

The per-network IGN losses only constrain f_A on A-images and on noise.
Nothing constrains it on f_B's output range. So the cheapest way to make
`f_A∘f_B ≈ f_B∘f_A` is for both to send those inputs to one shared image.

The distance metric only changes *which* image:
- **L1** rewards a blurry mean face.
- **LPIPS** rewards a sharp "mask" face (dark eyes, red nose and mouth) that
  AlexNet features agree on, with diversity down to 0.20 while sharpness still
  reads 0.43.

![LPIPS collapse](../results/method3_commuting/sweep/ms_lpips_lam5_lr1e-4_progression_real_3iter.png)

### Result 4: the classifier rewards collapse

λ = 20 at epoch 5 posts one of the sweep's highest out-of-distribution joint
rates, 0.55 / 0.57 at 2 / 3 iterations against naive's 0.50 / 0.51. Yet every
output is the same smudge (diversity 0.58, KID 16–20 vs naive's 5–7):

![λ=20 epoch 5, noise init](../results/method3_commuting/sweep/ms_lam20_e005_fftnoise.png)

So any joint-rate "gain" that comes with falling diversity is not a real gain.
The final comparison below therefore reads joint rate and KID together.

### Result 5: the one real gain — 1–5 epochs at λ = 5, lr = 2e-5

Lower lr slows the collapse enough that an early window opens up.

**Replication.** A second training seed was run with a snapshot every epoch.
Then the key checkpoints were re-evaluated on a **fresh, larger x₀** (1024
images, different seed) with KID. In the table below lower KID is better, and
KID is ×100.

| checkpoint | in-dist joint (1 / 2 / 3) | in-dist KID (1 / 2 / 3) | OOD joint (1 / 2 / 3) | OOD KID (1 / 2 / 3) | order dist |
|---|---|---|---|---|---|
| naive | 0.22 / 0.32 / 0.40 | 8.0 / 8.3 / 9.9 | 0.54 / 0.46 / 0.48 | 4.5 / 5.3 / 6.8 | 10.9 |
| λ = 0, lr = 2e-5, epoch 5 (control) | 0.21 / 0.29 / 0.35 | 8.2 / 10.5 / 13.4 | 0.47 / 0.36 / 0.39 | 4.9 / 6.6 / 9.8 | 12.3 |
| **λ = 5, lr = 2e-5, epoch 1** (seed 1) | **0.27 / 0.38 / 0.46** | **7.8 / 8.0 / 9.1** | 0.53 / 0.50 / 0.50 | 6.0 / 6.9 / 7.8 | 5.4 |
| λ = 5, lr = 2e-5, epoch 3 (seed 1) | 0.26 / 0.41 / 0.48 | 7.9 / 8.9 / 10.9 | 0.51 / 0.48 / 0.50 | 5.3 / 6.7 / 8.6 | 4.8 |
| λ = 5, lr = 2e-5, epoch 5 (seed 0) | 0.26 / 0.40 / 0.45 | 8.2 / 9.9 / 12.6 | 0.47 / 0.43 / 0.45 | 4.9 / 6.9 / 9.5 | 4.5 |
| λ = 5, lr = 2e-5, epoch 10 (seed 1) | 0.28 / 0.40 / 0.48 | 8.8 / 11.9 / 15.4 | 0.48 / 0.42 / 0.47 | 5.3 / 8.8 / 12.8 | 4.1 |
| λ = 1, lr = 2e-5, epoch 5 | 0.23 / 0.35 / 0.38 | 7.6 / 8.6 / 11.4 | 0.44 / 0.39 / 0.39 | 4.8 / 5.9 / 8.4 | 5.7 |
| LPIPS λ = 5, lr = 2e-5, epoch 5 | 0.24 / 0.38 / 0.46 | 8.2 / 10.6 / 13.2 | 0.48 / 0.50 / 0.53 | 5.6 / 7.8 / 10.4 | 5.8 |
| λ = 20, lr = 1e-4, epoch 5 | 0.28 / 0.38 / 0.46 | 12.7 / 18.0 / 21.2 | 0.52 / 0.48 / 0.56 | 8.5 / 16.0 / 20.3 | 2.7 |
| trained f_A∩B baseline | 0.58 | 6.6 | 0.91 | 3.4 | — |
| real Male ∧ Smiling photos (classifier ceiling) | 0.87 | — | 0.87 | — | — |

**In distribution, one epoch of commute fine-tuning is a clean improvement.**
- **Joint rate:** up +0.05–0.06 at every depth. Treating the two sets of
  1024 outputs as independent, that is about 2.5–3 standard errors. The
  inputs are actually paired, which would tighten this, but no paired test
  was run.
- **KID:** equal or slightly *better* than naive.
- **Order distance:** halved.
- **Controls:** the matched λ = 0 control shows no gain.
- **Both seeds agree:** seed 0 at epoch 5 and seed 1 at epochs 1–10 give the
  same joint rates.
- **After that, quality erodes:** KID climbs with every further epoch.

**Out of distribution it is a wash.** The joint rate is about the same as
naive, and KID is somewhat worse.

![best: λ=5, lr=2e-5, 1 epoch, real init](../results/method3_commuting/sweep/ms_best_lam5_lr2e-5_e001_real.png)
![best: epoch progression, noise init](../results/method3_commuting/sweep/ms_best_lam5_lr2e-5_progression_fftnoise_3iter.png)

### The full Male+Smiling sweep

All values below use the 256-image x₀. Each cell reads: in-dist joint (1 / 3
iter) | OOD joint (3 iter) | order dist | diversity (3 iter).

| config | epoch 5 | epoch 50 |
|---|---|---|
| naive | 0.22 / 0.42 \| 0.51 \| 10.9 \| 0.99 | |
| λ = 0, lr = 1e-4 | 0.21 / 0.37 \| 0.40 \| 12.4 \| 0.95 | 0.18 / 0.34 \| 0.34 \| 10.5 \| 0.93 |
| λ = 0, lr = 2e-5 | 0.19 / 0.41 \| 0.41 \| 12.3 \| 0.97 | 0.16 / 0.29 \| 0.34 \| 13.6 \| 0.96 |
| λ = 0.2, lr = 1e-4 | 0.20 / 0.36 \| 0.38 \| 6.7 \| 0.95 | 0.23 / 0.23 \| 0.29 \| 5.7 \| 0.90 |
| λ = 1, lr = 1e-4 | 0.18 / 0.33 \| 0.35 \| 5.4 \| 0.91 | 0.21 / 0.15 \| 0.17 \| 4.4 \| 0.76 |
| λ = 1, lr = 2e-5 | 0.24 / 0.39 \| 0.47 \| 5.7 \| 0.91 | 0.23 / 0.33 \| 0.38 \| 4.8 \| 0.81 |
| λ = 5, lr = 1e-4 | 0.28 / 0.32 \| 0.33 \| 4.1 \| 0.76 | 0.01 / 0.00 \| 0.00 \| 2.1 \| 0.27 |
| λ = 5, lr = 2e-5 | 0.27 / 0.48 \| 0.50 \| 4.5 \| 0.75 | 0.17 / 0.00 \| 0.00 \| 2.8 \| 0.38 |
| λ = 20, lr = 1e-4 | 0.27 / 0.50 \| 0.57 \| 2.7 \| 0.53 | 0.00 / 0.00 \| 0.00 \| 1.1 \| 0.15 |
| LPIPS λ = 5, lr = 1e-4 | 0.19 / 0.46 \| 0.52 \| 5.6 \| 0.66 | 0.14 / 0.02 \| 0.03 \| 4.5 \| 0.20 |
| LPIPS λ = 5, lr = 2e-5 | 0.22 / 0.48 \| 0.58 \| 5.8 \| 0.80 | 0.15 / 0.32 \| 0.44 \| 4.9 \| 0.44 |

The pattern across the whole sweep:
- Any λ ≥ 0.2 roughly halves order distance within 5 epochs.
- Higher λ or higher lr make the collapse come sooner.
- Lower lr is the knob that buys a usable early window.

![all configs, metrics vs epoch](../results/method3_commuting/sweep/ms_metrics_vs_epoch.png)
![all configs at epoch 50](../results/method3_commuting/sweep/ms_overview_real_3iter_e050.png)

Per-snapshot numbers are in
[ms_sweep_results.csv](../results/method3_commuting/sweep/ms_sweep_results.csv) and
[ms_robust_n1024_kid_results.csv](../results/method3_commuting/sweep/ms_robust_n1024_kid_results.csv).

### Male + Blond_Hair: no help, and the commute term changes which attribute wins

This is the anti-correlated pair. The classifier recognizes only **0.27** of
real male-blond photos as both, so joint rates here are small at best. The
table shows 3-iteration in-distribution rates.

| config | joint | Male | Blond | order dist | diversity |
|---|---|---|---|---|---|
| naive | 0.04 | 0.09 | 0.46 | 19.0 | 0.80 |
| λ = 0, lr = 2e-5, epoch 50 | 0.05 | 0.18 | 0.38 | 17.2 | 0.86 |
| λ = 1, lr = 2e-5, epoch 50 | 0.04 | 0.48 | 0.12 | 6.5 | 0.78 |
| λ = 1, lr = 1e-4, epoch 50 | 0.00 | 0.70 | 0.00 | 5.1 | 0.72 |
| λ = 5, lr = 1e-4, epoch 5 | 0.01 | 0.59 | 0.05 | 5.4 | 0.61 |
| λ = 5, lr = 1e-4, epoch 50 | 0.00 | 0.88 | 0.01 | 1.6 | 0.26 |

Naive alternation ends mostly blond and not male. With the commute term the
tug-of-war flips: Male wins and blond disappears. The pair commutes by one
projector yielding to the other, not by finding the intersection, and at
λ = 5 the same blur collapse follows.

![Male+Blond, all configs at epoch 50](../results/method3_commuting/sweep/mb_overview_real_3iter_e050.png)

### Correction to Part 1

That doc's order-distance table reads "Male+Smiling, real images: 7.0 → 3.5",
which looks like naive → commuting. It is actually the commuting-trained pair
at 20 iterations vs at 1 iteration (from `exp_results/Male_Smiling_commuting_eval*.log`).
Naive was never measured there. On the same x₀ naive is **10.9** (Male+Smiling)
and **19.0** (Male+Blond), so the drop from commute training is larger than
that doc implies: 10.9 → 3.5 and 19.0 → 4.4.

### Conclusion and what to try next

Training f_A and f_B to commute does make them commute. But with only the
per-network IGN losses as anchors, the commute term is satisfied mostly
through a degenerate shortcut: both compositions send each net's outputs to
one shared image.

The useful regime is a **very short, low-lr fine-tune: 1–5 epochs at λ = 5,
lr = 2e-5**. There it buys a real but modest in-distribution gain before the
shortcut takes over. More epochs are strictly worse. Even at its best, the
pair stays far from a network trained directly on the intersection
(0.27 / 0.38 / 0.46 vs 0.58).

Directions this points to, none of them tried yet:

- **Anchor the cross outputs.** Apply each net's IGN losses (idempotence +
  tightness) to the *other* net's outputs too, so that `f_A(f_B(x))` is held
  to f_A's own projection behaviour instead of being free to collapse.
- **Early-stop on diversity or KID**, not on the commute loss, which keeps
  falling straight through the collapse.
- **Rule out batch-statistics shortcuts.** The commute term sees batches of
  the other net's outputs, whose BatchNorm statistics differ from real images.
  Mixing real and composed images in one batch would test whether the nets are
  keying on that.

### Reproducing

```bash
# the sweep (trains + evaluates; resumable) -- ~90 s/epoch Male+Smiling, ~32 s/epoch Male+Blond on an A10
python -m experiments.method3_commuting.commuting_sweep --spec experiments/configs/commuting_sweep_male_smiling.json
python -m experiments.method3_commuting.commuting_sweep --spec experiments/configs/commuting_sweep_male_smiling_followup.json
python -m experiments.method3_commuting.commuting_sweep --spec experiments/configs/commuting_sweep_male_blond.json

# figures
python -m experiments.method3_commuting.make_commuting_grids --eval_dir exp_results/commuting_sweep/male_smiling --comparison_epochs 5 10 50
python -m experiments.method3_commuting.make_commuting_grids --eval_dir exp_results/commuting_sweep/male_blond \
    --pair_label "Male AND Blond_Hair" --comparison_epochs 5 10 50

# robustness pass: fresh 1024-image x0 + KID vs real Male-and-Smiling photos
python -m experiments.method3_commuting.commuting_sweep_eval --kid --num_samples 1024 --seed 1 \
    --naive_a runs/Male_500_withsamples/ckpt_0075.pt --naive_b runs/Smiling_paper500/ckpt_0075.pt \
    --baseline runs/Baseline_Male_AND_Smiling/ckpt_last.pt --intersection_spec "Male,Smiling" \
    --runs runs/ms_l1_lam0_lr2e-5 runs/ms_l1_lam5_lr2e-5 runs/ms_l1_lam5_lr2e-5_s1_fine ... \
    --epochs 1 2 3 5 10 --out exp_results/commuting_sweep/male_smiling_robust
```
