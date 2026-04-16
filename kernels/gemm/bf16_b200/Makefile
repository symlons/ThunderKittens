#GPU := B200
#SRC := bf16_b200_gemm.cu
#OUT := bf16_b200_gemm.out
#CMD := ./bf16_b200_gemm.out
#CONFIG := standalone
#include ../../common.mk


GPU := B200
SRC := bf16_b200_gemm.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python3 -c "import _C"
CONFIG := pytorch
include ../../common.mk
