#!/bin/bash
# Parameter Golf - 1xH100 Probe Experiments
# Run these in order on a 1xH100 RunPod instance.
# Each probe is ~3 minutes of training + eval time.
#
# Setup (run once):
#   cd /workspace && git clone <your-fork> parameter-golf && cd parameter-golf
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#
# Usage: bash run_probes.sh <probe_number>
#   e.g., bash run_probes.sh 0

set -euo pipefail

COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=180 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100"

PROBE=${1:-0}

case $PROBE in
  0)
    echo "=== Probe 0: 1xH100 Scaling Check ==="
    env $COMMON \
    RUN_ID=probe_scaling_check \
    SEED=1337 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_0_scaling.log
    ;;
  1)
    echo "=== Probe 1: Baseline Reproduction ==="
    env $COMMON \
    RUN_ID=probe_baseline_s1337 \
    SEED=1337 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_1_baseline.log
    ;;
  2)
    echo "=== Probe 2: Eval Seq Len 2048 ==="
    # Uses the same training as baseline but evals at 2048 tokens
    env $COMMON \
    RUN_ID=probe_eval2048_s1337 \
    SEED=1337 \
    EVAL_SEQ_LEN=2048 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_2_eval2048.log
    ;;
  3)
    echo "=== Probe 3: Extended Warmdown (3000 iters) ==="
    env $COMMON \
    RUN_ID=probe_warmdown3000_s1337 \
    SEED=1337 \
    WARMDOWN_ITERS=3000 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_3_warmdown3000.log
    ;;
  4)
    echo "=== Probe 4: SWA ==="
    env $COMMON \
    RUN_ID=probe_swa_s1337 \
    SEED=1337 \
    USE_SWA=1 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_4_swa.log
    ;;
  5)
    echo "=== Probe 5: SWA + Extended Warmdown ==="
    env $COMMON \
    RUN_ID=probe_swa_wd3000_s1337 \
    SEED=1337 \
    USE_SWA=1 \
    WARMDOWN_ITERS=3000 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_5_swa_warmdown3000.log
    ;;
  6)
    echo "=== Probe 6: QAT During Warmdown ==="
    env $COMMON \
    RUN_ID=probe_qat_s1337 \
    SEED=1337 \
    QAT_ENABLED=1 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_6_qat.log
    ;;
  7)
    echo "=== Probe 7: Full Combo (QAT + SWA + Warmdown 3000) ==="
    env $COMMON \
    RUN_ID=probe_combo_s1337 \
    SEED=1337 \
    QAT_ENABLED=1 \
    USE_SWA=1 \
    WARMDOWN_ITERS=3000 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_7_combo.log
    ;;
  8)
    echo "=== Probe 8: Full Combo + Weight Decay ==="
    env $COMMON \
    RUN_ID=probe_combo_wd_s1337 \
    SEED=1337 \
    QAT_ENABLED=1 \
    USE_SWA=1 \
    WARMDOWN_ITERS=3000 \
    WEIGHT_DECAY=0.02 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_8_combo_wd.log
    ;;
  9)
    echo "=== Probe 9: Best Combo x 3 Seeds ==="
    # UPDATE the overrides below with whatever won from probes 1-8
    for seed in 1337 42 7; do
      echo "--- Seed $seed ---"
      env $COMMON \
      RUN_ID=probe_best_s${seed} \
      SEED=${seed} \
      QAT_ENABLED=1 \
      USE_SWA=1 \
      WARMDOWN_ITERS=3000 \
      torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_9_best_s${seed}.log
    done
    ;;
  all)
    echo "=== Running All Probes Sequentially ==="
    mkdir -p logs
    for i in $(seq 0 8); do
      bash "$0" $i
      echo ""
    done
    echo "=== All probes complete. Run 'bash $0 9' for seed variance. ==="
    ;;
  *)
    echo "Usage: bash run_probes.sh {0|1|2|3|4|5|6|7|8|9|all}"
    exit 1
    ;;
esac

echo ""
echo "=== Done. Check final metrics with: ==="
echo "grep 'final_int8_zlib_roundtrip_exact' logs/probe_*.log"
