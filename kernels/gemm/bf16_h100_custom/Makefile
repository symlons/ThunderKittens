EXT := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

KERNELS := _C$(EXT) _gelu_bwd$(EXT) _linear_bwd$(EXT) _linear_bwd_fused$(EXT)

TORCH_LIB := $(shell python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
CUPTI_LIB := $(firstword $(wildcard /usr/local/cuda*/extras/CUPTI/lib64 /usr/libexec/datacenter-gpu-manager-4/plugins/cuda12))
PYTHON := env PYTHONPATH="$(CURDIR):$$PYTHONPATH" LD_LIBRARY_PATH="$(TORCH_LIB):$(CUPTI_LIB):$$LD_LIBRARY_PATH" python3

.PHONY: all bench test reports dashboard clean clean_reports

all: $(KERNELS)

_C$(EXT): bf16_h100_gemm.cu
	$(MAKE) -f Makefile.gemm BUILD_MODE=torch

_gelu_bwd$(EXT): gelu_bwd.cu
	$(MAKE) -f Makefile.gelu_bwd BUILD_MODE=torch

_linear_bwd$(EXT): linear_bwd.cu
	$(MAKE) -f Makefile.linear_bwd BUILD_MODE=torch

_linear_bwd_fused$(EXT): linear_bwd_fused.cu
	$(MAKE) -f Makefile.linear_bwd_fused BUILD_MODE=torch

bench: $(KERNELS)
	$(PYTHON) harness.py all --report KERNEL_REPORT.md

test: $(KERNELS)
	$(PYTHON) harness.py test --report KERNEL_REPORT.md

reports: bench test
	$(PYTHON) amdahls_analysis.py > AMDAHL_REPORT.txt

dashboard: reports
	$(PYTHON) generate_dashboard.py

clean:
	rm -f $(KERNELS) *.out *.ptx
	rm -rf __pycache__

clean_reports:
	rm -f KERNEL_REPORT.md BENCH_REPORT.md CORRECTNESS_AND_PROFILE_REPORT.md AMDAHL_REPORT.txt dashboard.html
