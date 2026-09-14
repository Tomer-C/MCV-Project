# IGNs for Projection onto Target-Space Intersections

Code for the MCV project *"Using IGNs For Projection Onto Target Space
Intersections Via the Alternating Projection Theorem"*.

Trains one Idempotent Generative Network per CelebA attribute, then applies them
alternately — `x_{n+1} = f_B(f_A(x_n))` — to test whether the iteration lands in
the intersection of their target sets, and compares that zero-shot result
against an IGN trained directly on `A ∩ B`.

Built on the reference implementation in `../IGN-main` (Shocher et al., ICLR
2024). The IGN losses are preserved exactly; see *Relationship to upstream*.

## Layout

| file | role |
|---|---|
| `ign_data.py` | CelebA loading + **attribute-subset filtering** (the `A`, `B`, `A∩B` splits) |
| `ign_models.py` | IGN architecture (DCGAN-style encoder/decoder) + attribute classifier |
| `ign_utils.py` | loss manager, DDP/device helpers, checkpointing |
| `train_ign.py` | trains **one** IGN on one attribute spec |
| `train_classifier.py` | trains the attribute classifier used to judge outputs |
| `alternating.py` | **the core contribution**: the alternating iteration + convergence diagnostics |
| `metrics.py` | attribute rates, FID/KID, LPIPS/SSIM/PSNR/L2 |
| `run_experiment.py` | runs Phases 2 & 4 and writes `results.json` |
| `make_figures.py` | all report figures from `results.json` |
| `scripts/inspect_attributes.py` | **run first**: attribute counts & intersection sizes |
| `scripts/smoke_test.py` | fast correctness checks, no data needed |
| `scripts/integration_test.py` | full pipeline on synthetic data, no data needed |
| `scripts/run_pipeline.sh` | end-to-end driver |

**New here?** [TRAINING.md](TRAINING.md) walks through training an IGN and
tuning the hyperparameters.

## Quick start

```bash
pip install torch torchvision matplotlib numpy
# optional: pip install lpips wandb gdown

python scripts/smoke_test.py          # verify the code (no CelebA needed)
python scripts/integration_test.py    # verify the pipeline  (no CelebA needed)
```

Then get CelebA (see `scripts/README_data.md`) and:

```bash
python scripts/inspect_attributes.py            # pick a viable feature pair
bash scripts/run_pipeline.sh                    # full experiment
ATTR_A=Male ATTR_B=Smiling NPROC=4 bash scripts/run_pipeline.sh   # 4 GPUs
```

## Running the phases manually

```bash
# Phase 1 — one IGN per feature
python train_ign.py --attr_spec "Smiling"    --run_name Smiling    --num_epochs 60
python train_ign.py --attr_spec "Eyeglasses" --run_name Eyeglasses --num_epochs 60

# Phase 3 — the baseline trained directly on the intersection
python train_ign.py --attr_spec "Smiling,Eyeglasses" --run_name Smiling_AND_Eyeglasses --num_epochs 60

# evaluation instrument
python train_classifier.py --epochs 8

# Phases 2 & 4 — alternate, compare, test order sensitivity
python run_experiment.py \
    --models runs/Smiling/ckpt_last.pt runs/Eyeglasses/ckpt_last.pt \
    --names f_A f_B \
    --baseline runs/Smiling_AND_Eyeglasses/ckpt_last.pt \
    --intersection_spec "Smiling,Eyeglasses" \
    --num_iters 20 --num_samples 2048

python make_figures.py --results results/Smiling_AND_Eyeglasses
```

Multi-GPU: prefix any training command with
`torchrun --nproc_per_node=8` and add `--ddp`.

### Attribute specs

`--attr_spec` accepts comma-separated CelebA attribute names; prefix with `-`
to require *absence*:

```
"Smiling"                smiling
"Smiling,Eyeglasses"     smiling AND wearing glasses      <- an intersection
"Smiling,-Male"          smiling AND not male
"all"                    unfiltered
```

### Scaling past two features

Everything generalises to `K` models — pass more checkpoints:

```bash
python run_experiment.py \
  --models runs/Smiling/ckpt_last.pt runs/Eyeglasses/ckpt_last.pt runs/Male/ckpt_last.pt \
  --names f_A f_B f_C \
  --baseline runs/Smiling_AND_Eyeglasses_AND_Male/ckpt_last.pt \
  --intersection_spec "Smiling,Eyeglasses,Male"
```

All `K!` orders are evaluated automatically (6 for K=3).

## How each proposal question is answered

| Proposal question | Where | Output |
|---|---|---|
| Can an IGN project onto an intersection? (contribution 1) | `train_ign.py` with a multi-attribute spec | `attr_rates.baseline` — how often `f_A∩B` output has both features |
| Does alternating converge? | `alternating.py` → `step_norm` | `figures/convergence.png`; `alternating.converged` |
| Does it diverge / zigzag? (challenge 2) | `cycle_distance`, `image_norm` | `figures/convergence.png`; `alternating.diverged` |
| Is `x*` in the intersection? | `metrics.attr_rates` | `attr_rates.alternating.joint` |
| Does `x*` match `f_A∩B(x_0)`? (Phase 4) | `metrics.paired_distances` | `paired_vs_baseline` (L2/PSNR/SSIM/LPIPS) |
| Are outputs realistic? | `metrics.frechet_distance` | `distribution.*.fid` |
| Does order matter? | `alternating.run_all_orders` | `orders`, `order_disagreement` |
| Enough data for the baseline? (challenge 1) | `scripts/inspect_attributes.py` | intersection counts before training |

### Reading the diagnostics

`alternating.py` records, per iteration:

- **`step_norm`** `‖x_{n+1}−x_n‖` — shrinking toward 0 ⇒ convergence.
- **`idem_residual_i`** `‖f_i(x_n)−x_n‖` — distance to each model's fixed-point
  set. **All of them small at once is the operational definition of "in the
  intersection".**
- **`cycle_distance`** `min_{m<n}‖x_n−x_m‖` — near 0 while `step_norm` stays
  large ⇒ a limit cycle (the "zigzagging" the proposal warns about), not
  convergence.
- **`image_norm`** — growing without bound ⇒ numerical explosion.

If the undamped iteration oscillates, `damped_alternating_projection`
(`x_{n+1} = (1−α)x_n + α f(x_n)`) is the standard remedy and is run
automatically; compare in `figures/damping.png`.

## Interpreting the headline number

`attr_rates.*.joint` is the fraction of outputs the classifier says carry **all**
target attributes. Compare four rows:

- `x0` — the input (chance level)
- `single_f_A` / `single_f_B` — one IGN alone; should get *its own* attribute
  right and the other one wrong
- `alternating` — **the method under test**
- `baseline` — the directly trained `f_A∩B`
- `real_intersection` — real `A∩B` photos; the classifier's ceiling, since it is
  not a perfect judge

The claim "alternating projection reaches the intersection zero-shot" is
supported to the extent `alternating` approaches `baseline` and
`real_intersection`, while clearly exceeding both single models.

## Caveats worth stating in the report

- **FID needs samples.** InceptionV3 features are 2048-d; with fewer samples
  than that the covariance is rank-deficient and FID is strongly biased.
  Use `--num_samples 2048` minimum (10000 ideally) for reported numbers.
  `metrics.py` prints a warning when the estimate is unreliable.
- **KID can be negative.** It is the unbiased MMD² estimator, so
  same-distribution samples fluctuate around zero. Not a bug; compare KIDs
  against each other.
- **The classifier is not ground truth.** Always report
  `real_intersection` alongside, as the achievable ceiling.
- **Von Neumann's theorem does not apply here.** It assumes orthogonal
  projections onto convex sets. IGNs are non-linear, oblique, and only
  approximately idempotent, so convergence is an empirical question — which is
  precisely what this code measures. The proposal already frames the theorem as
  inspiration only; keep that framing.

## Relationship to upstream (`IGN-main`) and the paper

Preserved exactly:
- the three IGN losses, including which copy of `f` receives the gradient
  (`f_copy` for `loss_out`, detached `f_z` for `loss_out_d`), and the tanh
  clamp on the discriminative term;
- the DCGAN-enhanced 64×64 architecture (`--arch dcgan_enhanced`) and its
  weight init.

Paper-faithful defaults (note: Shocher's released `run_celeba.sh` and his
paper's Appendix C **disagree**; the defaults follow the paper):
- loss weights `20 / 20 / 2.5` with clamp ratio `1.5` (the script used
  `1 / 1 / 10` with clamp ratio `1.0` — pass those flags to reproduce it);
- `--noise_type x_fft_stats`: noise with the data's per-frequency statistics,
  as the paper uses for CelebA (the script used plain white noise);
- `--arch dcgan`: the plain 5-layer DCGAN autoencoder the paper describes
  (the script pointed at a `MUNet` whose file is missing from the skeleton;
  the skeleton's `dcgan.py` has a channel mismatch and cannot run);
- sample grids and experiments run with **train-mode BatchNorm** (batch
  statistics), the only mode upstream ever uses — an IGN's running stats are
  polluted by the noise/generated batches, so eval-mode outputs can look
  collapsed even when the model is healthy (`run_experiment.py --bn_mode eval`
  restores the old behaviour);
- training checkpoints now carry the noise frequency statistics, so
  `run_experiment.py --init noise` samples matched noise; old checkpoints fall
  back to white noise with a warning, and `--arch` is read from the checkpoint
  (old ones load as `dcgan_enhanced`).

Changed, and why:
- **attribute filtering** — upstream has no notion of subsets; this is the
  project's core requirement;
- **`SyncBatchNorm` → `resolve_norm()`** — `SyncBatchNorm` raises without an
  initialised process group, so upstream cannot run off a multi-GPU cluster.
  Now auto-selects `SyncBatchNorm` under DDP, `BatchNorm2d` otherwise;
- **`DistributedSampler` made conditional** — same reason;
- **wandb optional** — upstream hard-codes an entity name and crashes offline;
- **added** inference/checkpoint loading, the alternating module, metrics, and
  figures — none of which exist upstream.
