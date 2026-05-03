# Unified build — single command rebuilds all kernels.
#
# Usage:
#   make all         # build all PyTorch .so extensions
#   make bench       # run benchmarks + generate BENCH_REPORT.md
#   make test        # run correctness tests + CORRECTNESS_AND_PROFILE_REPORT.md
#   make reports     # bench + test + Amdahl analysis
#   make dashboard   # reports + HTML dashboard
#   make clean       # remove build artifacts
#   make clean_reports  # remove generated reports

EXT := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

KERNELS := _C$(EXT) _gelu_bwd$(EXT) _linear_bwd$(EXT) _linear_bwd_fused$(EXT)

# Set up environment for running Python scripts
PYTHON := env PYTHONPATH="$(CURDIR):$$PYTHONPATH" LD_LIBRARY_PATH="$$(python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")" python3

.PHONY: all bench test reports dashboard clean clean_reports

all: $(KERNELS)

# Each sub-Makefile sets SRC/OUT/CMD/CONFIG and includes ../../common.mk

_C$(EXT): bf16_h100_gemm.cu
	$(MAKE) -f Makefile.gemm BUILD_MODE=torch

_gelu_bwd$(EXT): gelu_bwd.cu
	$(MAKE) -f Makefile.gelu_bwd BUILD_MODE=torch

_linear_bwd$(EXT): linear_bwd.cu
	$(MAKE) -f Makefile.linear_bwd BUILD_MODE=torch

_linear_bwd_fused$(EXT): linear_bwd_fused.cu
	$(MAKE) -f Makefile.linear_bwd_fused BUILD_MODE=torch

bench: $(KERNELS)
	$(PYTHON) harness.py bench --report BENCH_REPORT.md

test: $(KERNELS)
	$(PYTHON) harness.py test --report CORRECTNESS_AND_PROFILE_REPORT.md

reports: bench test
	$(PYTHON) amdahls_analysis.py > AMDAHL_REPORT.txt

dashboard: reports
	$(PYTHON) generate_dashboard.py

clean:
	rm -f $(KERNELS) *.out *.ptx
	rm -rf __pycache__

clean_reports:
	rm -f BENCH_REPORT.md CORRECTNESS_AND_PROFILE_REPORT.md AMDAHL_REPORT.txt dashboard.html
