# Configuration: set BUILD_MODE=torch for PyTorch .so or BUILD_MODE=cuda for standalone executable
# Usage: make BUILD_MODE=torch   OR   make BUILD_MODE=cuda

BUILD_MODE ?= cuda

GPU := H100
SRC := bf16_h100_gemm.cu

ifeq ($(BUILD_MODE),torch)
	OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
	CMD :=
	CONFIG := pytorch
	NVCCFLAGS := -DTORCH_COMPILE
else ifeq ($(BUILD_MODE),cuda)
	OUT := bf16_h100_gemm.out
	CMD := ./bf16_h100_gemm.out
	CONFIG := standalone
else
	$(error Invalid BUILD_MODE. Use 'torch' or 'cuda')
endif

include ../../common.mk
