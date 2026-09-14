# Training an IGN: how-to and hyperparameter guide

This is the practical companion to the [README](README.md): how to launch a
training run, what the outputs mean, and which knobs to turn when something
looks wrong.

## 1. Prerequisites

```bash
pip install torch torchvision matplotlib numpy
```

- **CelebA** at `./data/celeba/` (see `scripts/README_data.md`; torchvision's
  auto-download is often rate-limited, the manual route is more reliable).
- Optional, both improve monitoring and degrade gracefully when absent:
  - `runs/classifier/classifier.pt` — the attribute classifier
    (`python train_classifier.py --epochs 8`) enables the per-epoch
    confidence curves;
  - `attr_splits/` — ground-truth has/lacks id lists
    (`python make_attr_splits.py`) speed up the per-epoch feature check.

Sanity-check the install without any data:

```bash
python scripts/smoke_test.py
python scripts/integration_test.py
```

## 2. Training

One IGN per attribute subset. Basic run (all defaults are the paper's):

```bash
python train_ign.py --attr_spec "Smiling" --run_name Smiling_500 --num_epochs 500
```

- `--attr_spec` accepts comma-separated CelebA attributes; prefix `~` (or `-`)
  for absence: `"Male"`, `"~Young"`, `"Smiling,Eyeglasses"`, `"Smiling,~Male"`.
- Resume: `--resume runs/Smiling_500/ckpt_last.pt` (raise `--num_epochs` to
  extend a finished run; loss weights are re-read from the command line, the
  optimizer state comes from the checkpoint).
- Multi-GPU: prefix with `torchrun --nproc_per_node=N` and add `--ddp`.
- Long remote runs: `nohup python train_ign.py ... > run.log 2>&1 &`.

### What appears in `runs/<run_name>/` while training

| output | meaning |
|---|---|
| `samples/eNNNN.png` | rows: fixed FFT noise z, f(z), f²(z), f³(z); fresh noise + same; real x, f(x) |
| `feature_check/*_eNNNN.png` | 10 has + 10 lacks faces (fixed & fresh, train & test splits) with f applied — the projection view |
| `loss_curves.png` | the three loss components, weighted total, held-out test reconstruction, classifier confidence |
| `loss_history.json`, `loss_steps.csv` | raw per-epoch / per-step loss data |
| `ckpt_NNNN.pt`, `ckpt_last.pt` | weights + optimizer, every `--save_every` (default 25) epochs |

## 3. The hyperparameters

Defaults follow the paper (Appendix C). Note the paper and the released
`run_celeba.sh` **disagree**; the script's values are listed as the alternative.

| flag | default (paper) | run_celeba.sh | what it does |
|---|---|---|---|
| `--in_manifold_loss_factor` | 20 | 1 | weight of reconstruction `\|f(x)-x\|` |
| `--out_of_manifold_loss_factor` | 20 | 1 | weight of idempotence `\|f(f(z))-f(z)\|` (inner f) |
| `--discriminative_loss_factor` | 2.5 | 10 | weight of the tightness term (outer f, adversarial-like) |
| `--discriminative_loss_clamp_ratio` | 1.5 | 1.0 | tanh-clamp of tightness at ratio × loss_in |
| `--noise_type` | `x_fft_stats` | `normal` | FFT frequency-matched noise vs white gaussian |
| `--arch` | `dcgan` | (MUNet, unavailable) | plain paper autoencoder; `dcgan_enhanced` = residual variant |
| `--batch_size` | 256 | 256 (×8 GPUs) | larger = smoother BatchNorm statistics |
| `--lr`, `--beta1`, `--beta2` | 1e-4, 0.5, 0.999 | same | Adam; constant, no scheduler |
| `--loss_p` | 1.0 | 1.0 | L1 distance |

## 4. Tuning: what to watch, and which knob to turn

Open `loss_curves.png` and the `feature_check/test_*` grids. The panels to
watch are **test recon** (generalizing reconstruction), **dP has** (does f
preserve the feature on images that have it — should stay ≈ 0) and
**P(feat) after f on lacks** (does f *add* the feature — the projection
strength).

| symptom | likely cause | knob |
|---|---|---|
| outputs collapse to one repeated blob/texture; f(f(z)) goes black | tightness term dominating | lower `--discriminative_loss_factor` (toward 2.5) and/or `--discriminative_loss_clamp_ratio` |
| reconstruction great, but lacks-images pass through unchanged (weak projection) | tightness too weak relative to recon; over-training | raise `--discriminative_loss_factor` (5–10, keep the clamp), or use an earlier checkpoint (see below) |
| grids look collapsed but losses look fine | you are rendering with eval-mode BatchNorm | don't — all provided scripts use train-mode (batch statistics), the only mode the reference code ever uses |
| everything blurry | early epochs / small subset | train longer; or compose an early "projector" checkpoint with a late "sharpener" one |
| loss_in flat but samples still improving | normal | judge by grids, not L1 |

Two empirical findings baked into this repo, worth knowing before tuning:

1. **Projection strength peaks early and decays.** As reconstruction sharpens,
   off-manifold images gradually become fixed points and f stops changing
   them. The best *projector* checkpoint is typically epoch ~50–100, not
   `ckpt_last`. Sweep your checkpoints with:
   ```bash
   python rebuild_loss_curves.py --run_dir runs/Smiling_500     # per-ckpt metrics
   python make_feature_grids.py  --run_dir runs/Smiling_500     # per-ckpt grids
   ```
   and pick the checkpoint maximizing "P after f on lacks" at acceptable recon.
2. **Transfer strength scales with how far the lacks-set is from the manifold.**
   Global attributes (Male, Young, hair color) project far better than small
   local edits (Smiling). Choose features accordingly.

For experiments that need both strong transfer and sharpness, combine
checkpoints instead of retraining (early = projector, late = sharpener):

```bash
python cross_checkpoint_experiments.py --run_a runs/Male_500 --run_b runs/Not_Young_500
```

## 5. Evaluating a trained model

```bash
# f applied to with/without-feature images, train & test splits, + P(feature)
python apply_to_negatives.py --feature Smiling --ckpt runs/Smiling_500/ckpt_0075.pt

# the full alternating-projection experiment between two trained IGNs
python run_experiment.py \
    --models runs/A/ckpt_0075.pt runs/B/ckpt_0075.pt --names f_A f_B \
    --baseline runs/A_AND_B/ckpt_last.pt --intersection_spec "A,B"
```
