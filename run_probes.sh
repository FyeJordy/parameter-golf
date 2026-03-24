#!/bin/bash
# Parameter Golf - Wave 1 competitive probes on 1xH100
# Usage: bash run_probes.sh {0|1|2|3|4|5|6|all}

set -euo pipefail

mkdir -p logs

COMMON="DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=180 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100"

BASELINE_OVERRIDES="NUM_LAYERS=9 MLP_MULT=2 TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=1200 \
PAIRHASH_ENABLED=0 EMA_ENABLED=0 EVAL_STRIDE=0 EVAL_DOC_ISOLATED=0 \
EXPORT_MODE=int8 USE_ZSTD=0 QAT_ENABLED=0 USE_SWA=0"

ARCH_OVERRIDES="NUM_LAYERS=11 MLP_MULT=3 TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=1200 \
PAIRHASH_ENABLED=1 PAIRHASH_BUCKETS=8192 PAIRHASH_DIM=96 \
EMA_ENABLED=0 EVAL_STRIDE=0 EVAL_DOC_ISOLATED=0 \
EXPORT_MODE=int8 USE_ZSTD=0 QAT_ENABLED=0 USE_SWA=0"

TRAINING_OVERRIDES="NUM_LAYERS=11 MLP_MULT=3 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 WARMDOWN_ITERS=3500 \
PAIRHASH_ENABLED=1 PAIRHASH_BUCKETS=8192 PAIRHASH_DIM=96 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
EVAL_STRIDE=0 EVAL_DOC_ISOLATED=0 \
EXPORT_MODE=int8 USE_ZSTD=0 QAT_ENABLED=1 USE_SWA=0"

COMPRESS_OVERRIDES="NUM_LAYERS=11 MLP_MULT=3 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 WARMDOWN_ITERS=3500 \
PAIRHASH_ENABLED=1 PAIRHASH_BUCKETS=8192 PAIRHASH_DIM=96 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
EVAL_STRIDE=0 EVAL_DOC_ISOLATED=0 \
EXPORT_MODE=int8 USE_ZSTD=1 QAT_ENABLED=1 USE_SWA=0"

FULL_WAVE1_OVERRIDES="NUM_LAYERS=11 MLP_MULT=3 TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 WARMDOWN_ITERS=3500 \
PAIRHASH_ENABLED=1 PAIRHASH_BUCKETS=8192 PAIRHASH_DIM=96 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
EVAL_STRIDE=64 EVAL_DOC_ISOLATED=1 \
EXPORT_MODE=int8 USE_ZSTD=1 QAT_ENABLED=1 USE_SWA=0"

PROBE=${1:-0}

case $PROBE in
  0)
    echo "=== Probe 0: Wave 1 Scaling Check ==="
    env $COMMON \
    RUN_ID=probe_wave1_scaling \
    SEED=1337 \
    $FULL_WAVE1_OVERRIDES \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_0_wave1_scaling.log
    ;;
  1)
    echo "=== Probe 1: Old Baseline Reference ==="
    env $COMMON \
    RUN_ID=probe_baseline_reference \
    SEED=1337 \
    $BASELINE_OVERRIDES \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_1_baseline.log
    ;;
  2)
    echo "=== Probe 2: Architecture Only ==="
    env $COMMON \
    RUN_ID=probe_architecture \
    SEED=1337 \
    $ARCH_OVERRIDES \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_2_architecture.log
    ;;
  3)
    echo "=== Probe 3: + Training Stack ==="
    env $COMMON \
    RUN_ID=probe_training_stack \
    SEED=1337 \
    $TRAINING_OVERRIDES \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_3_training.log
    ;;
  4)
    echo "=== Probe 4: + Compression Stack ==="
    env $COMMON \
    RUN_ID=probe_compression_stack \
    SEED=1337 \
    $COMPRESS_OVERRIDES \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_4_compression.log
    ;;
  5)
    echo "=== Probe 5: + Eval Stack ==="
    env $COMMON \
    RUN_ID=probe_eval_stack \
    SEED=1337 \
    $FULL_WAVE1_OVERRIDES \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_5_eval.log
    ;;
  6)
    echo "=== Probe 6: Best Wave 1 x 3 Seeds ==="
    for seed in 1337 42 7; do
      echo "--- Seed $seed ---"
      env $COMMON \
      RUN_ID=probe_best_wave1_s${seed} \
      SEED=${seed} \
      $FULL_WAVE1_OVERRIDES \
      torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/probe_6_best_s${seed}.log
    done
    ;;
  all)
    echo "=== Running Wave 1 Probes 0-5 ==="
    for i in 0 1 2 3 4 5; do
      bash "$0" "$i"
      echo ""
    done
    echo "=== Wave 1 probes complete. Run 'bash $0 6' for the 3-seed check. ==="
    ;;
  *)
    echo "Usage: bash run_probes.sh {0|1|2|3|4|5|6|all}"
    exit 1
    ;;
esac

echo ""
echo "=== Done. Check final metrics with: ==="
echo "grep 'final_export_roundtrip_exact' logs/probe_*.log"
echo "grep 'Total submission size export' logs/probe_*.log"
