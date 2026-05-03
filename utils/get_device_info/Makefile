GPU := H100
CONFIG := standalone
COMMON_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../kernels)
THUNDERKITTENS_ROOT := $(abspath $(COMMON_DIR)/..)

NVCC := nvcc
NVCCFLAGS := -std=c++20 -O3 --use_fast_math
NVCCFLAGS += -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -forward-unknown-to-host-compiler
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills
NVCCFLAGS += -I$(THUNDERKITTENS_ROOT)/include -I$(THUNDERKITTENS_ROOT)/prototype
NVCCFLAGS += -lineinfo -ftemplate-backtrace-limit=0

ifeq ($(GPU),B300)
  NVCCFLAGS += -DKITTENS_BLACKWELL -gencode arch=compute_103a,code=sm_103a
else ifeq ($(GPU),B200)
  NVCCFLAGS += -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else ifeq ($(GPU),H100)
  NVCCFLAGS += -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a
else ifeq ($(GPU),A100)
  NVCCFLAGS += -DKITTENS_AMPERE -gencode arch=compute_80,code=sm_80
else
  $(error Unsupported GPU: $(GPU). Set GPU to H100, B200, B300, or A100.)
endif

.PHONY: all run clean device_info async_engines

all: device_info async_engines

device_info:
	$(NVCC) device_info.cu $(NVCCFLAGS) -o device_info.out

async_engines:
	$(NVCC) async_engines.cu $(NVCCFLAGS) -o async_engines.out

run: all
	./device_info.out
	./async_engines.out

clean:
	rm -f device_info.out async_engines.out
