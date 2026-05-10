# You must set the environment variable ARCH
# export ARCH=SM90 | SM100

ifeq ($(ARCH),SM100)
SRC := gemm_rs_b200.cu
else ifeq ($(ARCH),SM90)
SRC := gemm_rs_h100.cu
else
$(error Unsupported ARCH: $(ARCH). Please set ARCH to SM90 or SM100.)
endif

OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 benchmark.py
CONFIG := pytorch

include ../../common.mk
