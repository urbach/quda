include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I.

HDRS = blas_reference.h wilson_dslash_reference.h    \
	test_util.h dslash_util.h

TESTS = blas_test $(WILSON_DIRAC_TEST)		

all: $(TESTS)

wilson_dslash_test: wilson_dslash_test.o test_util.o wilson_dslash_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

wilson_invert_test: wilson_invert_test.o test_util.o wilson_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

blas_test: blas_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	-rm -f *.o wilson_dslash_test wilson_invert_test blas_test

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
