"""
Deprecated: cuBLAS GEMM benchmarks are now registered in harness.py.

    python3 harness.py bench cublas_dW cublas_dx cublas_ab   # individual
    python3 harness.py bench                                   # all (includes cuBLAS)
"""
import sys
import subprocess

print("DEPRECATED: bench_cublas_gemms.py is now integrated into harness.py")
print("Run:  python3 harness.py bench cublas_dW cublas_dx cublas_ab")
sys.exit(subprocess.run(["python3", "harness.py", "bench", "cublas_dW", "cublas_dx", "cublas_ab"]).returncode)
