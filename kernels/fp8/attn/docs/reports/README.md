# Generated Reports

This directory contains generated correctness and quantization report snapshots.

Regenerate them from the repository root with:

```bash
python3 correctness_quant.py
python3 correctness_attn.py
python3 correctness_realistic.py --unet none
HF_HOME=/scratch python3 correctness_realistic.py --unet THUDM/CogVideoX-2b \
  --out docs/reports/QUANT_REALISTIC_COGVIDEOX.md
```

The default `--out` paths for the correctness scripts already point here.
