#define REG_CREATE(x, value) QudaSumFloat x = value
#define SH_SET(s, i, t) s[i] = t
#define REDUCE(x, i) x += REDUCE_OPERATION(i)
#define SH_SUM(s, i, j) s[i] += s[j]
#define AUXILIARY(i) REDUCE_AUXILIARY(i);
#define SUMFLOAT_P(x, y) QudaSumFloat *s = y;
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a = b
#define WRITE_GLOBAL(x, i, s, j) x[i] = s[j]

#include "reduce_core.h"

template <typename Float>
double REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {
  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
  // when there is only one warp per block, we need to allocate two warps 
  // worth of shared memory so that we don't index shared memory out of bounds
  size_t smemSize = blasBlock.x * sizeof(QudaSumFloat) * (blasBlock.x <= 32 ? 2 : 1);

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

#undef SH_SUM
#undef SH_SET
#undef REG_CREATE
#undef REDUCE

#undef AUXILIARY
#undef SUMFLOAT_P
#undef WRITE_GLOBAL
#undef SUMFLOAT_EQ_SUMFLOAT
