"""
Deprecated: GELU backward functionality now lives in harness.py.

    python3 harness.py test gelu         # correctness
    python3 harness.py bench             # benchmark (includes gelu components)
"""
import sys
import subprocess

print("DEPRECATED: test_gelu_bwd.py is now integrated into harness.py")
print("Run:  python3 harness.py test gelu")
sys.exit(subprocess.run(["python3", "harness.py", "test", "gelu"]).returncode)