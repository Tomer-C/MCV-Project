# Experimental Results

All experiments target the proposal's question: can images with **multiple
specific features** be synthesized by combining single-feature IGNs
(alternating-projection style), and how does that compare to an IGN trained
directly on the intersection? Everything below is zero-shot (no intersection
training) unless it says "baseline". Curated figures and raw numbers live in
[`results/`](results/); the scripts that produced them are in the repo (see
*Reproducing* at the bottom).

Feature pair for the headline numbers: **Male ∧ Smiling** (CelebA, 64×64).
Judge: the attribute classifier (`train_classifier.py`); realism: FID on
Inception features vs real test-set intersection images (1024 samples —
biased in absolute terms, comparable across rows).

## Headline table

| method | joint rate | FID | kept | where |
|---|---|---|---|---|
| **Trained baseline** `f_Male∩Smiling` | 0.91 | 51 | 100% | [`results/baseline_comparison/`](results/baseline_comparison/) |
| **Adaptive relaxed stepping** (zero-shot) | 0.67 | 50 | 100% | [`results/recipes/adaptive_stepping.json`](results/recipes/adaptive_stepping.json) |
| **Relaxed alternation** α=0.3 (zero-shot) | 0.55 | 52 | 100% | [`results/recipes/relaxed_alternation.json`](results/recipes/relaxed_alternation.json) |
| **Generate-and-select** (zero-shot) | 1.00* | 47 | 55% | [`results/recipes/league.json`](results/recipes/league.json) |
| Ordinary alternation (best tuning) | 0.63 | 146 | 100% | `results/recipes/league.json` (`c4`) |
| Naive alternation (matched stages, 15 iters) | 0.42 | 159 | 100% | `results/baseline_comparison/compare_noise.json` |

\* joint = 1.00 by construction (the selector is the judge); the independent
evidence is FID, the diversity score, and the grids. Numbers are fresh-seed
confirmed (3 seeds, [`fresh_seed_confirmation.json`](results/recipes/fresh_seed_confirmation.json));
the tuning-seed values were ~0.03–0.05 higher (winner's curse, measured).

**Bottom line:** the trained baseline wins the joint-attribute rate
decisively, but zero-shot pipelines built purely from single-feature
components reach **baseline-level realism** (FID parity) at 55–74% of its
joint rate — or a certified 100% joint at 55% sampling yield. See
[`alt_vs_baseline_noise.png`](results/baseline_comparison/alt_vs_baseline_noise.png)
for the visual comparison (rows: noise → alternation → baseline).

## The three zero-shot recipes (how they work)

All start from **FFT-matched noise**: random noise whose per-frequency
mean/variance equal those of the target intersection's training images
(`--noise_stats_spec "Male,Smiling"`). No image content — only spectral
statistics.

1. **Relaxed alternation.** One full pass of the *sharp* projector
   (`f_Male` at its projector checkpoint, epoch 75), then a **partial** step
   of the fragile feature: `x ← x + α·(f_Smiling@25(x) − x)`, α ≈ 0.3, ONE
   cycle. The partial step is the classical relaxed (Krasnoselskii–Mann) form
   of alternating projections: (1−α) of the sharp image survives untouched,
   which removes the blur that full composition accumulates (~25 FID per
   pass) at zero cost in joint rate. Grid:
   [`relaxed_a0.3_random8.png`](results/recipes/relaxed_a0.3_random8.png).

2. **Adaptive relaxed stepping** (best selection-free result). Same as (1),
   but the smile step is split into small α=0.2 nudges and **each image stops
   as soon as the classifier scores its smile > 0.5** (max 4 steps; mean 1.6).
   Nothing is discarded — the classifier only allocates per-image effort.
   Dominates every fixed-α point on both axes: 0.67 @ FID 50. Grid:
   [`adaptive_stepping_random8.png`](results/recipes/adaptive_stepping_random8.png).

3. **Generate-and-select** (rejection sampling). Generate ~2× candidates with
   (1), keep every candidate the classifier certifies on **both** features
   (threshold only — never rank: ranking selects a homogeneous mode; see
   [`selection_top32_vs_random32.png`](results/recipes/selection_top32_vs_random32.png),
   top rows = ranked favorites, bottom rows = honest random members). The
   kept set matches the baseline's diversity (0.955 vs 0.958 relative to real
   photos, [`selection_diversity.json`](results/recipes/selection_diversity.json))
   at FID 47.

## Empirical laws found

1. **Projection strength peaks early in training and decays.** As
   reconstruction sharpens, off-manifold images become fixed points and the
   IGN stops moving them. Use ~epoch 50–100 checkpoints as projectors, not
   the last one (per-network curves: [`results/per_network/`](results/per_network/)).
   Exception: `Black_Hair` stays flat — re-applicable color edits don't get
   unlearned.
2. **One cycle is optimal; more cycles only blur.** The joint rate is an
   inverted U over iterations (peak ~1–4), and each extra composition pass
   costs ~25 FID. The von Neumann limit picture gives way to early stopping.
3. **Order: the sharp net generates, the fragile feature goes last/relaxed.**
   Reversing (Smiling first) costs 2× FID at equal joint
   ([`reversed_order.json`](results/limits/reversed_order.json)).
   The full order study is in [ORDER_EXPERIMENT.md](ORDER_EXPERIMENT.md). From noise, the
   first net decides the face and the two orders never meet. From real faces, order barely
   matters. Which order wins on joint rate depends on the checkpoint, and with the final
   checkpoints it is Male first at every cycle count.
4. **Feature compatibility decides everything.** Correlated pairs work
   (Male+Smiling, Male+Black: [`black_pairs_relaxed.json`](results/limits/black_pairs_relaxed.json));
   anti-correlated pairs fail under every method tried — `f_Blond` literally
   re-feminizes what `f_Male` masculinizes, watchable in
   [`male_blond_tug_of_war_trajectory.png`](results/limits/male_blond_tug_of_war_trajectory.png).
5. **Component quality is a hard floor.** The proposal's own example pair
   (smiling + glasses) fails not from incompatibility but because a ~10k-image
   IGN cannot render thin frames ([`glasses_pairs.json`](results/limits/glasses_pairs.json)).
6. **Classifier metrics diverge from human judgment.** The classifier's
   optimum (4 cycles) is blurrier than the eye's (1–2); always inspect
   *random*-sample grids, never top-scored ones.

## Scaling past two features (proposal phase: "upscale")

| K | features | raw joint | selected | yield | where |
|---|---|---|---|---|---|
| 2 | Male∧Smiling | 0.55 | 1.00 @ FID 47 | 0.56 | `results/recipes/` |
| 3 | +Black_Hair | 0.31 | 1.00 @ FID 61 | 0.29 | [`results/scaling/triple.json`](results/scaling/triple.json) |
| 4 | +Young | 0.21 | 1.00 @ FID 60 | 0.22 | [`results/scaling/quad.json`](results/scaling/quad.json) |
| 3 | with **Blond** (adversarial) | **0.004–0.016** | — | — | [`k3_with_blond_failure.json`](results/scaling/k3_with_blond_failure.json) |

Graceful yield decay for compatible sets; one adversarial member kills the
whole iteration.

These K=3/4 numbers use **relaxed** steps. Plain cyclic alternation
(full-strength steps) does not extend: the per-order, per-checkpoint and
real-face study is in [MULTI_CLASS_EXPERIMENT.md](MULTI_CLASS_EXPERIMENT.md).

## Other findings

- **Damping the classic iteration is unnecessary** — no oscillation occurs;
  α=1 converges fastest in residual ([`damping.png`](results/limits/damping.png)).
  (Relaxation helps *realism*, which is a different mechanism.)
- **Real-image editing trade-off**: relaxed steps preserve identity almost
  perfectly but under-deliver the features (joint ≤ 0.10); full-strength
  alternation delivers ~0.4 joint at identity cost. An identity↔attribute
  dial with no free point ([`real_image_editing_grid.png`](results/limits/real_image_editing_grid.png)).
- **Heavy-tightness fine-tuning does not restore projection strength** once
  the fixed-point set has broadened (tested: resume Male@75 with the
  tightness weight at 10 for 40 epochs — stable training, no transfer gain).
- Single-network projection quality ranking: Male ≫ Black_Hair ≈ Blond >
  Smiling ≫ Young (see `results/per_network/`, e.g.
  [`Male_negatives_test.png`](results/per_network/Male_negatives_test.png)).

## Reproducing

Experiments run on a GPU machine via two scripts (credentials are NOT in the
repo — copy `vm_config.template.json` to `vm_config.json`, fill it in; it is
gitignored):

```bash
python vm_orchestrate.py run --skip_training   # push code, run suite, pull results
python vm_orchestrate.py start --spec adaptive_b10.json --status_file status_b10.json
python vm_orchestrate.py watch                 # poll + pull images/reports (never weights)
```

`vm_experiments.py` is the on-VM runner (the standard suite, fastest first);
each `adaptive_b*.json` is one ad-hoc batch exactly as executed during the
campaign (batches 2/3/8 contain `<ABS_PROJECT_DIR>` placeholders — substitute
your VM project path). Dedicated evaluators: `league_eval.py` (recipe
league), `relax_eval.py` / `nosel_eval.py` (relaxation + adaptive stepping),
`diversity_eval.py`, `confirm_eval.py` (fresh seeds), `pairs_relax_eval.py`
(any pair via a JSON config), `triple_eval.py` / `quad_eval.py` (scaling), `multi_class_experiment.py`
(plain K-way alternation, every order),
`real_edit_eval.py`, `polish_eval.py`. Trained runs referenced:
`Male_500_withsamples`, `Smiling_paper500`, `Blond_Hair_500`, `Young_500`,
`Black_Hair_200`, `Eyeglasses_200`, `Baseline_Male_AND_Smiling` (weights stay
on the training machine).
