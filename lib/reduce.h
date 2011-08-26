#if NREDUCE == 1

#define REG_CREATE(x, value) QudaSumFloat x = value
#define SH_SET(s, i, t) s[i] = t
#define REDUCE(x, i) x += REDUCE_OPERATION(i)
#define SH_SUM(s, i, j) s[i] += s[j]
#define AUXILIARY(i) REDUCE_AUXILIARY(i);
#define SUMFLOAT_P(x, y) QudaSumFloat *s = y;
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a = b
#define WRITE_GLOBAL(x, i, s, j) x[i] = s[j]

#elif NREDUCE == 2

#define REG_CREATE(x, value) QudaSumFloat x##_r = value, x##_i = value
#define SH_SET(s, j, t) s##_r[j] = t##_r, s##_i[j] = t##_i
#define REDUCE(x, j) x##_r += REDUCE_REAL_OPERATION(j), x##_i += REDUCE_IMAG_OPERATION(j)
#define SH_SUM(s, j, k) s##_r[j] += s##_r[k], s##_i[j] += s##_i[k]
#define AUXILIARY(i) REDUCE_REAL_AUXILIARY(i); REDUCE_IMAG_AUXILIARY(i)
#define SUMFLOAT_P(x, y) QudaSumFloat *x##_r = y, *x##_i = y + reduce_threads
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a##_r = b##_r, a##_i = b##_i
#define WRITE_GLOBAL(array, j, s, k) array[j] = s##_r[k], array[j+gridDim.x] = s##_i[k]

#else

#define REG_CREATE(s, value) QudaSumFloat s##_x = value, s##_y = value, s##_z = value
#define SH_SET(s, i, t) s##_x[i] = t##_x, s##_y[i] = t##_y, s##_z[i] = t##_z
#define REDUCE(s, i)							\
  s##_x += REDUCE_X_OPERATION(i), s##_y += REDUCE_Y_OPERATION(i), s##_z += REDUCE_Z_OPERATION(i)
#define SH_SUM(s, i, j) s##_x[i] += s##_x[j], s##_y[i] += s##_y[j], s##_z[i] += s##_z[j]
#define AUXILIARY(i) REDUCE_X_AUXILIARY(i); REDUCE_Y_AUXILIARY(i); REDUCE_Z_AUXILIARY(i)
#define SUMFLOAT_P(s, t) QudaSumFloat *s##_x = t, *s##_y = t + reduce_threads, \
    *s##_z = t + 2*reduce_threads
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a##_x = b##_x, a##_y = b##_y, a##_z = b##_z 
#define WRITE_GLOBAL(array, i, s, j)					\
  array[i] = s##_x[j], array[i+gridDim.x] = s##_y[j], array[i+2*gridDim.x] = s##_z[j]
#endif

#include "reduce_core.h"

template <int N, typename R, typename FloatN>
R REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {
  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
  // when there is only one warp per block, we need to allocate two warps 
  // worth of shared memory so that we don't index shared memory out of bounds
  size_t smemSize = blasBlock.x * N*sizeof(QudaSumFloat) * (blasBlock.x <= 32 ? 2 : 1);

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
  cudaMemcpy(h_reduce, d_reduce, N*blasGrid.x*sizeof(QudaSumFloat), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();

  R sum_h;
  double *sum_p = (double*)&sum_h; // cast return type as an array of doubles
  for (unsigned int j=0; j<N; j++) sum_p[j] = 0.0;
  for (unsigned int i = 0; i < blasGrid.x; i++) 
    for (unsigned int j=0; j<N; j++) sum_p[j] += h_reduce[i + j*blasGrid.x];
  reduceDoubleArray(sum_p, N);
  return sum_h;
}

#undef SH_SUM
#undef SH_SET
#undef REG_CREATE
#undef REDUCE

#undef AUXILIARY
#undef SUMFLOAT_P
#undef WRITE_GLOBAL
#undef SUMFLOAT_EQ_SUMFLOAT
