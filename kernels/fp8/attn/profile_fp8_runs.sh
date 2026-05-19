#!/usr/bin/env bash
set -euo pipefail

# Short / moderate context: canonical 500/100 protocol via profile_fp8.py.
python3 profile_fp8.py --B 1 --H 8  --N 1536 --D 128 --seed 0 "$@"
python3 profile_fp8.py --B 1 --H 8  --N 3072 --D 128 --seed 0 "$@"
python3 profile_fp8.py --B 2 --H 16 --N 3072 --D 128 --seed 0 "$@"

# Long context (N ≥ 57600) across B ∈ {1, 2, 4}. Uses the same
# fp8_suite.profiling helpers (uniform_tensor, recommended_group_count,
# benchmark_ms with 2 CUDA events around the full launch loop), but
# defaults to warmup=50/iters=30 because per-iter time is 20-900 ms here
# (canonical 500/100 takes ~6 h end-to-end). Set PYTORCH_CUDA_ALLOC_CONF
# for stable allocator behaviour at high N.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 profile_long_context.py --seed 0 "$@"
