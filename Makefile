CUFILES		:= dslash_cuda.cu blas_cuda.cu
CCFILES		:= inv_bicgstab_cuda.cpp inv_cg.cpp util_cuda.cpp gauge_cuda.cpp spinor_quda.cpp

CUDA_INSTALL_PATH = /usr/local/cuda
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/sdk/C/common/inc
LIB = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

DFLAGS = #-D__DEVICE_EMULATION__

OPT=-O3
CC = gcc
CFLAGS = -Wall ${OPT} -std=c99 $(INCLUDES) ${DFLAGS}
CXX = g++
CXXFLAGS = -Wall ${OPT} $(INCLUDES) ${DFLAGS} 
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc 
NVCCFLAGS = ${OPT} $(INCLUDES) ${DFLAGS} -arch=sm_13 # --maxrregcount=64 #-deviceemu
LDFLAGS = -fPIC $(LIB)
CCOBJECTS = $(CCFILES:.cpp=.o)
CUOBJECTS = $(CUFILES:.cu=.o)
DEPS = $(wildcard *.d)
TARGETS=dslash_test invert_test su3_test pack_test llfat_test  gauge_force_test fermion_force_test

default: all
all: ${TARGETS}

ILIB = libquda.a
ILIB_OBJS = inv_cg_quda.o dslash_quda.o blas_quda.o util_quda.o \
	dslash_reference.o blas_reference.o invert_quda.o gauge_quda.o spinor_quda.o misc.o llfat_reference.o  gauge_force_reference.o fermion_force_reference.o hw_quda.o
ILIB_DEPS = $(ILIB_OBJS) blas_quda.h quda.h util_quda.h invert_quda.h gauge_quda.h spinor_quda.h enum_quda.h dslash_reference.h misc.h llfat_quda.h  llfat_reference.h gauge_force_quda.h  gauge_force_reference.h kernel_common_macro.h fermion_force_quda.h fermion_force_reference.h hw_quda.h

$(ILIB): $(ILIB_DEPS)
	ar cru $@ $(ILIB_OBJS)

invert_test: invert_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@

dslash_test: dslash_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@

su3_test: su3_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@

pack_test: pack_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@

llfat_test: llfat_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@ 

gauge_force_test: gauge_force_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@

fermion_force_test: fermion_force_test.o $(ILIB)
	$(CXX) $(LDFLAGS) $< $(ILIB) -o $@

clean:
	-rm -f *.o $(ILIB) *.d ${TARGETS}
deepclean: clean
	rm -f *~
cubin: dslash_quda.cu
	${NVCC} --ptx $(NVCCFLAGS) $<
	ptxas -arch=sm_13 dslash_quda.ptx -o dslash_quda.cubin
	decuda.py -A -k _Z30testSiteComputeGenStapleKernelP6float2S0_P6float4S2_S0_S0_ dslash_quda.cubin
%.o: %.c
	$(CC) -MD $(CFLAGS) $< -c -o $@

%.o: %.cpp
	$(CXX) -MD $(CXXFLAGS) $< -c -o $@

%.o: %.cu 
	$(NVCC) -M $(INCLUDES) ${DFLAGS} $<  > $*.d
	$(NVCC)  $(NVCCFLAGS) $< -c -o $@

ifneq "${DEPS}" ""
include ${DEPS}
endif

