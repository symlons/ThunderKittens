#!/usr/bin/env bash
set -euo pipefail

python3 profile_fp8.py --B 1 --H 8  --N 1536 --D 128 --seed 0 "$@"
python3 profile_fp8.py --B 1 --H 8  --N 3072 --D 128 --seed 0 "$@"
python3 profile_fp8.py --B 2 --H 16 --N 3072 --D 128 --seed 0 "$@"
