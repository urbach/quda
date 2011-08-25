#if (REDUCE_TYPE == REDUCE_KAHAN) // Kahan compensated summation

#define SH_STRIDE 2 // stride is two elements
#define SH_SUM(s, i, j) dsadd(s[i], s[i+1], s[i], s[i+1], s[2*j], s[2*j+1]) 
#define SH_SET(s, i, x) s[i] = x##0, s[i+1] = x##1
#define SH_EVAL(s, i) s[i] + s[i+1]
#define REG_CREATE(x, value) QudaSumFloat x##0 = value, x##1 = value
#define REDUCE(x, i) dsadd(x##0, x##1, x##0, x##1, REDUCE_OPERATION(i), 0)

#else // Regular summation

#define SH_STRIDE 1
#define SH_SUM(s, i, j) s[i] += s[j]
#define SH_SET(s, i, x) s[i] = x
#define SH_EVAL(s, i) s[i]
#define REG_CREATE(x, value) QudaSumFloat x = value
#define REDUCE(x, i) x += REDUCE_OPERATION(i)

#endif

#define AUXILIARY(i) REDUCE_AUXILIARY(i);
#define SUMFLOAT_P(x, y) QudaSumFloat *s = y;
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a = b
#define WRITE_GLOBAL(x, i, s, j) x[i] = SH_EVAL(s,j)

#include "reduce_core.h"

template <typename Float>
double REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {
  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
  // when there is only one warp per block, we need to allocate two warps 
  // worth of shared memory so that we don't index shared memory out of bounds
#if (REDUCE_TYPE == REDUCE_KAHAN)
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 4 * sizeof(QudaSumFloat) : blasBlock.x * 2 * sizeof(QudaSumFloat);
#else
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 2 * sizeof(QudaSumFloat) : blasBlock.x * sizeof(QudaSumFloat);
#endif

  if (blasBlock.x == 32) {
    REDUCE_FUNC_NAME(Kernel)<32><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduce, n);
  } else if (blasBlock.x == 64) {
    REDUCE_FUNC_NAME(Kernel)<64><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduce, n);
  } else if (blasBlock.x == 128) {
    REDUCE_FUNC_NAME(Kernel)<128><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduce, n);
  } else if (blasBlock.x == 256) {
    REDUCE_FUNC_NAME(Kernel)<256><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduce, n);
  } else if (blasBlock.x == 512) {
    REDUCE_FUNC_NAME(Kernel)<512><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduce, n);
  } else if (blasBlock.x == 1024) {
    REDUCE_FUNC_NAME(Kernel)<1024><<< blasGrid, blasBlock, smemSize >>>(REDUCE_PARAMS, d_reduce, n);
  } else {
    errorQuda("Reduction not implemented for %d threads", blasBlock.x);
  }

  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduce, d_reduce, blasGrid.x*sizeof(QudaSumFloat), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();

  double cpu_sum = 0;
  for (unsigned int i = 0; i < blasGrid.x; i++) cpu_sum += h_reduce[i];

  reduceDouble(cpu_sum);

  return cpu_sum;
}

#undef SH_STRIDE
#undef SH_SUM
#undef SH_SET
#undef SH_EVAL
#undef REG_CREATE
#undef REDUCE

#undef AUXILIARY
#undef SUMFLOAT_P
#undef WRITE_GLOBAL
#undef SUMFLOAT_EQ_SUMFLOAT
