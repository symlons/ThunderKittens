#GPU := H100
#SRC := bf16_h100_gemm.cu
#OUT := bf16_h100_gemm.out
#CMD := ./bf16_h100_gemm.out
#CONFIG := standalone
#include ../../common.mk


GPU := H100
SRC := bf16_h100_gemm.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD :=
CONFIG := pytorch
include ../../common.mk
