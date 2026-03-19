# Parameter Golf — Session Handoff

## Status: Phase 2 COMPLETE, Ready for Phase 3 (GPU Testing)

All code modifications to `train_gpt.py` are implemented, syntax-verified, and code-reviewed. Three code review findings have been fixed. The code is ready for 1xH100 probe experiments.

---

## What's Been Done

### Phase 1 ✅ — Environment Setup & Baseline Verification
- Python venv created (but MLX testing limited by system Python 3.9 — needs 3.10+)
- `BASELINE_REFERENCE.md` created with all extracted metrics
- Baseline: post-quant val_loss=2.0727, val_bpb=1.2244, artifact=15,863,489 bytes

### Phase 2 ✅ — All Code Modifications Implemented in `train_gpt.py`

| Modification | Lines | Env Var | Status |
|---|---|---|---|
| SWA (Stochastic Weight Averaging) | Hyperparams ~90, training loop ~1066-1080, post-training ~1132-1145 | `USE_SWA=1 SWA_EVERY_N_STEPS=10` | ✅ Done |
| QAT (Quantization-Aware Training) | `fake_quantize_per_row()` ~357-374, CastedLinear ~545-555, loop toggle ~1066 | `QAT_ENABLED=1` | ✅ Done |
| Extended Warmdown | No code change needed — already env-configurable | `WARMDOWN_ITERS=3000` | ✅ Done |
| Weight Decay (Adam only) | optimizer_tok ~915, optimizer_scalar ~930, optimizer_head ~939 | `WEIGHT_DECAY=0.02` | ✅ Done |
| Eval Sequence Length | eval_val() uses `eval_sl`, val_tokens loading | `EVAL_SEQ_LEN=2048` | ✅ Done |
| torch.compile guard | Line ~887: `torch._dynamo.config.guard_nn_modules = True` | - | ✅ Done |
| Scale dtype match in QAT | Line ~371: `scale.to(torch.float16).to(torch.float32)` | - | ✅ Done |

### MLX mirror (`train_gpt_mlx.py`)
- SWA and weight decay mirrored for local testing
- QAT NOT mirrored (requires torch.quantile)

### Automation Scripts Created
- `run_probes.sh` — 10 probe experiments for 1xH100 (usage: `bash run_probes.sh {0-9|all}`)
- `run_full_validation.sh` — 5-seed validation for 8xH100

### Submission Template Created
- `records/track_10min_16mb/2026-03-XX_SWA_QAT_Warmdown/` with README.md and submission.json

---

## What's Next

### Phase 3 — 1xH100 Probe Experiments (~6 GPU-hours, ~$15-20)

**Setup on RunPod 1xH100:**
```bash
cd /workspace && git clone <your-fork> parameter-golf && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
mkdir -p logs
```

**Run probes in order:**
```bash
bash run_probes.sh 0   # Scaling check (MUST pass: step time <500ms, val_loss <3.0)
bash run_probes.sh 1   # Baseline reproduction
bash run_probes.sh 2   # Eval seq_len=2048
bash run_probes.sh 3   # Extended warmdown (WARMDOWN_ITERS=3000)
bash run_probes.sh 4   # SWA only
bash run_probes.sh 5   # SWA + warmdown
bash run_probes.sh 6   # QAT only
bash run_probes.sh 7   # Full combo (QAT + SWA + WARMDOWN=3000)
bash run_probes.sh 8   # Full combo + weight decay
bash run_probes.sh 9   # Best combo × 3 seeds (variance check)
```

**Decision tree:**
- If probe 5 (SWA + warmdown) alone beats 0.005 nats → skip QAT
- If probe 7 needed → include QAT
- If probe 8 shows additional gain → include weight decay
- If probe 9 variance too high → need 5+ seeds in Phase 4

**Key metric to check after each probe:**
```bash
grep 'final_int8_zlib_roundtrip_exact' logs/probe_*.log
grep 'Total submission size int8' logs/probe_*.log
```

### Phase 4 — Full 8xH100 Validation (5 seeds, ~$60)
```bash
# Edit BEST_OVERRIDES in run_full_validation.sh with winning probe config
bash run_full_validation.sh
```

### Phase 5 — Package Submission
1. Fill in `submission.json` with actual metrics
2. Fill in `README.md` multi-seed table
3. Copy best seed's `train.log` to submission folder
4. Open PR to upstream repo

---

## Key Technical Details for Next Session

### The quantization gap is the #1 leverage point
- Baseline quant gap: 0.0072 bpb (pre 1.2172 → post 1.2244)
- 4-hour run quant gap GROWS to 0.0325 bpb — more training = worse quantization
- Our interventions (SWA, QAT) directly reduce this gap

### Environment variables for the best config
```bash
USE_SWA=1
SWA_EVERY_N_STEPS=10
QAT_ENABLED=1
WARMDOWN_ITERS=3000
# Optionally: WEIGHT_DECAY=0.02, EVAL_SEQ_LEN=2048, QAT_USE_AMAX=1 (perf fallback)
```

### Target
- Must beat: post-quant val_loss < 2.0677 (≥0.005 nats improvement)
- With p < 0.01 across 5 seeds (one-sided t-test)

### Files modified
- `train_gpt.py` — all modifications (SWA, QAT, weight decay, eval seq_len, torch.compile guard)
- `train_gpt_mlx.py` — SWA + weight decay mirror only
- New files: `BASELINE_REFERENCE.md`, `run_probes.sh`, `run_full_validation.sh`, submission template

### Potential issues to watch for
1. **QAT step time**: If `torch.quantile` per-row slows steps >15%, set `QAT_USE_AMAX=1` as fallback
2. **SWA memory**: Weights are cloned to CPU (`detach().cpu().clone()`) — should be fine for 17M params
3. **torch.compile recompilation**: `guard_nn_modules=True` causes one recompile when QAT activates at warmdown start — expect a ~10s pause mid-training

### Plan file
Full plan: `~/.claude/plans/misty-fluttering-pinwheel.md`
