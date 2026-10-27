#!/bin/bash
# End-to-end driver for the two-feature experiment.
#
#   bash scripts/run_pipeline.sh                     # defaults: Smiling x Eyeglasses
#   ATTR_A=Male ATTR_B=Smiling bash scripts/run_pipeline.sh
#   NPROC=4 bash scripts/run_pipeline.sh             # multi-GPU via torchrun
#
# Set NPROC>1 to train each IGN with torchrun across that many GPUs.
set -e

# Defaults tuned for strong, reportable results. Override any via env var.
# Male x Smiling has a large A-and-B intersection so the baseline f_AnB trains
# well (Smiling x Eyeglasses was too rare -- check with inspect_attributes.py).
ATTR_A=${ATTR_A:-Male}
ATTR_B=${ATTR_B:-Smiling}
EPOCHS=${EPOCHS:-100}       # lower idempotence residual -> cleaner fixed points
BATCH=${BATCH:-128}         # tuned default; 256 is fine on a 48GB GPU
NPROC=${NPROC:-1}
DATA_ROOT=${DATA_ROOT:-./data}
NUM_ITERS=${NUM_ITERS:-40}  # headroom to see step_norm plateau or a limit cycle
NUM_SAMPLES=${NUM_SAMPLES:-2048}  # FID min for a non-degenerate covariance; 10000 for final
CLS_EPOCHS=${CLS_EPOCHS:-12}      # the classifier is the judge -- train it well

INTERSECTION="${ATTR_A},${ATTR_B}"
SLUG_A=$ATTR_A
SLUG_B=$ATTR_B
SLUG_AB="${ATTR_A}_AND_${ATTR_B}"

if [ "$NPROC" -gt 1 ]; then
  LAUNCH="torchrun --nproc_per_node=$NPROC --master_port ${PORT:-1236}"
  DDP_FLAG="--ddp"
else
  LAUNCH="python"
  DDP_FLAG=""
fi

echo "===========================PART 2============================="
echo " A = $ATTR_A     B = $ATTR_B     (intersection: $INTERSECTION)"
echo " epochs=$EPOCHS batch=$BATCH nproc=$NPROC"
echo "=============================================================="

echo -e "\n### Steps 2 & 4: alternating projection + comparison"
python -m experiments.method1_single_ign.run_experiment \
    --models "runs/$SLUG_A/ckpt_last.pt" "runs/$SLUG_B/ckpt_last.pt" \
    --names f_A f_B \
    --baseline "runs/$SLUG_AB/ckpt_last.pt" \
    --intersection_spec "$INTERSECTION" \
    --classifier runs/classifier/classifier.pt \
    --num_iters "$NUM_ITERS" --num_samples "$NUM_SAMPLES" \
    --data_root "$DATA_ROOT" --tag "$SLUG_AB"

echo -e "\n### Figures"
python -m experiments.method1_single_ign.make_figures --results "results/$SLUG_AB"

echo -e "\nDone. See results/$SLUG_AB/"
