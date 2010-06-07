###### Local configuration:

CUDA_INSTALL_PATH = /usr/local/cuda

CPU_ARCH = x86_64   # x86 or x86_64
GPU_ARCH = sm_20    # sm_10, sm_11, sm_12, sm_13 or sm_20
GPU_EMU  = false    # set to 'true' for device emulation

PYTHON = python     # python 2.5 or later required for 'make gen'
DEVICE = 0          # CUDA device to use for 'make tune'

######

INC = -I$(CUDA_INSTALL_PATH)/include

ifeq ($(strip $(CPU_ARCH)), x86_64)
  LIB = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart # for release 2.3 and later
  # LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart # for release 2.2 and earlier
  COPT =
else
  LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart
  COPT = -malign-double
endif

ifeq ($(strip $(GPU_EMU)), true)
  COPT += -D__DEVICE_EMULATION__
  NVCCOPT = -deviceemu
endif

ifeq ($(strip $(GPU_ARCH)), sm_20)
	GPU_ARCH_FLAGS= -D__CUDA_ARCH__=200 -ftz=true -prec-div=false -prec-sqrt=false 
else
	GPU_ARCH_FLAGS= -D__CUDA_ARCH__=130
endif

CC = gcc
CFLAGS = -O3 -Wall  -std=c99 $(COPT) $(INC) -DCUDA_VERSION=3000
CXX = g++
CXXFLAGS = -O3 -Wall  $(COPT) $(INC) -DCUDA_VERSION=3000
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc 
NVCCFLAGS = -O3 $(NVCCOPT) -arch=$(GPU_ARCH) $(INC) --ptxas-options=-v -DCUDA_VERSION=3000 ${GPU_ARCH_FLAGS}
LDFLAGS = -fPIC $(LIB)
MPICXX=mpicxx
