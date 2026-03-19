# Baseline Reference Numbers (from 8xH100 train.log)

## Model
- Parameters: 17,059,912
- Architecture: 9 layers, 512 dim, 1024 vocab, 8 heads, 4 KV heads, 2x MLP, tied embeddings

## Training
- Steps completed: 13,780 / 20,000 (stopped by 600s wallclock cap)
- Step average: 43.54ms
- Tokens/step: 524,288
- Total tokens: 7,224,688,640 (7.22B)
- World size: 8 GPUs
- Warmdown started at: ~step 12,580 (at ~548s)
- Warmdown iters: 1,200

## Warmdown Analysis (the key leverage point)
| Step | val_loss | val_bpb | Time (ms) | Phase |
|------|----------|---------|-----------|-------|
| 12400 | 2.1001 | 1.2406 | 539,897 | Pre-warmdown |
| 12600 | 2.0998 | 1.2404 | 548,606 | Warmdown start |
| 12800 | 2.0930 | 1.2364 | 557,317 | Warmdown |
| 13000 | 2.0859 | 1.2322 | 566,032 | Warmdown |
| 13200 | 2.0790 | 1.2281 | 574,747 | Warmdown |
| 13400 | 2.0718 | 1.2239 | 583,458 | Warmdown |
| 13600 | 2.0649 | 1.2197 | 592,172 | Warmdown |
| 13780 | 2.0606 | 1.2172 | 600,038 | End |

Warmdown improved val_bpb by 0.0232 in ~1200 steps (0.00002 bpb/step).

## Artifact
- Code: 47,642 bytes
- Model (int8+zlib): 15,815,847 bytes
- Total: 15,863,489 bytes
- Headroom: 136,511 bytes
- Compression ratio: 3.91x

## Scores
- Pre-quant: val_loss=2.0606, val_bpb=1.2172
- Post-quant: val_loss=2.0727, val_bpb=1.2244
- Quant gap: 0.0121 nats / 0.0072 bpb

## Target
- Must beat: val_loss < 2.0677 (improve by >= 0.005 nats)
- With p < 0.01 across multiple seeds
