BUILD_MODE ?= cuda

GPU := H100
SRC := fp8_h100_fwd.cu

ifeq ($(BUILD_MODE),torch)
	OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
	CMD :=
	CONFIG := pytorch
	NVCCFLAGS := -DTORCH_COMPILE -UTORCH_EXTENSION_NAME -DTORCH_EXTENSION_NAME=_C
else ifeq ($(BUILD_MODE),cuda)
	OUT := fp8_h100_fwd.out
	CMD := ./fp8_h100_fwd.out
	CONFIG := standalone
else
	$(error Invalid BUILD_MODE. Use 'torch' or 'cuda')
endif

include ../../common.mk

ifeq ($(BUILD_MODE),torch)
PYTORCH_LIBDIR_PATH ?= $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])")
NVCCFLAGS += -Xlinker -rpath -Xlinker $(PYTORCH_LIBDIR_PATH)
endif
