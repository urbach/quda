include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I.

HDRS = blas_reference.h twisted_mass_dslash_reference.h test_util.h

TESTS = twisted_mass_ndeg_dslash_test twisted_mass_ndeg_invert_test

all: $(TESTS)

twisted_mass_ndeg_dslash_test: twisted_mass_ndeg_dslash_test.o test_util.o twisted_mass_dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

twisted_mass_ndeg_invert_test: twisted_mass_ndeg_invert_test.o test_util.o twisted_mass_dslash_reference.o \
	        blas_reference.o $(QUDA)
	        $(CXX) $(LDFLAGS) $^ -o $@

clean:
	-rm -f *.o $(TESTS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

.PHONY: all clean
