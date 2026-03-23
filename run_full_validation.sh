#!/bin/bash
# Parameter Golf - Full 8xH100 Validation Runs
# Run this on an 8xH100 SXM instance after probe experiments determine the best config.
#
# Usage: bash run_full_validation.sh
#
# UPDATE THE OVERRIDES BELOW with your winning probe config before running!

set -euo pipefail
mkdir -p logs/full

# ============================================================
# EDIT THESE OVERRIDES with your best probe configuration
# ============================================================
BEST_OVERRIDES="USE_SWA=1 WARMDOWN_ITERS=3000 QAT_ENABLED=1"
# ============================================================

COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50"

echo "=== Full 8xH100 Validation ==="
echo "Config: $BEST_OVERRIDES"
echo ""

for seed in 1337 42 7 123 456; do
  echo "--- Seed $seed ---"
  env $COMMON $BEST_OVERRIDES \
  RUN_ID=full_best_s${seed} \
  SEED=${seed} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/full/full_s${seed}.log
  echo ""
done

echo ""
echo "=== All seeds complete ==="
echo ""
echo "Final metrics:"
grep 'final_export_roundtrip_exact' logs/full/full_s*.log
echo ""
echo "Submission sizes:"
grep 'Total submission size export' logs/full/full_s*.log
echo ""
echo "Run the significance check:"
echo "python3 -c \""
echo "from scipy import stats; import numpy as np"
echo "baseline_loss = 2.0727"
echo "results = []  # paste val_loss values from above"
echo "improvement = baseline_loss - np.mean(results)"
echo "se = np.std(results, ddof=1) / np.sqrt(len(results))"
echo "t_stat = (improvement - 0.005) / se"
echo "p_value = 1 - stats.t.cdf(t_stat, df=len(results)-1)"
echo "print(f'Mean improvement: {improvement:.6f} nats, p-value: {p_value:.6f}')"
echo "\""
