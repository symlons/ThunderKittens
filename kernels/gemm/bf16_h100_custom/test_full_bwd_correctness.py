"""
Deprecated: Full backward correctness is now registered in harness.py.

    python3 harness.py test full              # all 5 baselines + cross-comparison
    python3 harness.py test full --report REPORT.md
"""
import sys
import subprocess

print("DEPRECATED: test_full_bwd_correctness.py is now integrated into harness.py")
print("Run:  python3 harness.py test full")
sys.exit(subprocess.run(["python3", "harness.py", "test", "full"]).returncode)
