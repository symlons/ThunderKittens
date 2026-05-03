"""
Deprecated: Linear backward benchmarks are now registered in harness.py.

    python3 harness.py bench custom_bwd_unfused            # unfused backward
    python3 harness.py bench custom_bwd_fused              # fused backward
    python3 harness.py test linear fused full              # correctness
"""
import sys
import subprocess

print("DEPRECATED: test_linear_bwd.py is now integrated into harness.py")
print("Run:  python3 harness.py bench custom_bwd_unfused custom_bwd_fused")
sys.exit(subprocess.run(["python3", "harness.py", "bench", "custom_bwd_unfused", "custom_bwd_fused"]).returncode)
