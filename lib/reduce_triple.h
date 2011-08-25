#if (REDUCE_TYPE == REDUCE_KAHAN)

#define SH_STRIDE 2
#define REG_CREATE(s, value) QudaSumFloat s##0_x = value, s##1_x = value, s##0_y = value, s##1_y = value, \
    s##0_z = value, s##1_z = value
#define REDUCE(s, i) dsadd(s##0_x, s##1_x, s##0_x, s##1_x, REDUCE_X_OPERATION(i), 0); \
  dsadd(s##0_y, s##1_y, s##0_y, s##1_y, REDUCE_Y_OPERATION(i), 0) \
  dsadd(s##0_z, s##1_z, s##0_z, s##1_z, REDUCE_Z_OPERATION(i), 0)   
#define SH_SUM(s, i, j) dsadd(s##_x[i], s##_x[i+1], s##_x[i], s##_x[i+1], s##_x[2*j], s##_x[2*j+1]); \
  dsadd(s##_y[i], s##_y[i+1], s##_y[i], s##_y[i+1], s##_y[2*j], s##_y[2*j+1]); \
  dsadd(s##_z[i], s##_z[i+1], s##_z[i], s##_z[i+1], s##_z[2*j], s##_z[2*j+1]);
#define SH_SET(s, i, t) s##_x[i] = t##0_x, s##_x[i+1] = t##1_x, s##_y[i] = t##0_y, s##_y[i+1] = t##1_y \
    s##_z[i] = t##0_z, s##_z[i+1] = z##1_t
#define SH_EVAL(s, i) s[i] + s[i+1]

#else

#define SH_STRIDE 1
#define REG_CREATE(s, value) QudaSumFloat s##_x = value, s##_y = value, s##_z = value
#define REDUCE(s, i)							\
  s##_x += REDUCE_X_OPERATION(i), s##_y += REDUCE_Y_OPERATION(i), s##_z += REDUCE_Z_OPERATION(i)
#define SH_SUM(s, i, j) s##_x[i] += s##_x[j], s##_y[i] += s##_y[j], s##_z[i] += s##_z[j]
#define SH_SET(s, i, t) s##_x[i] = t##_x, s##_y[i] = t##_y, s##_z[i] = t##_z
#define SH_EVAL(s, i) s[i]

#endif

#define AUXILIARY(i) REDUCE_X_AUXILIARY(i); REDUCE_Y_AUXILIARY(i); REDUCE_Z_AUXILIARY(i)
#define SUMFLOAT_P(s, t) QudaSumFloat *s##_x = t, *s##_y = t + SH_STRIDE*reduce_threads, \
    *s##_z = t + 2*SH_STRIDE*reduce_threads
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a##_x = b##_x, a##_y = b##_y, a##_z = b##_z 
#define WRITE_GLOBAL(array, i, s, j)					\
  array[i] = SH_EVAL(s##_x, j), array[i+gridDim.x] = SH_EVAL(s##_y, j), array[i+2*gridDim.x] = SH_EVAL(s##_z, j)

#include "reduce_core.h"

template <typename Float2>
double3 REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {

  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_triple_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
#if (REDUCE_TYPE == REDUCE_KAHAN)
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 4 * sizeof(QudaSumFloat3) : blasBlock.x * 2 * sizeof(QudaSumFloat3);
#else
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 2 * sizeof(QudaSumFloat3) : blasBlock.x * sizeof(QudaSumFloat3);
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
  cudaMemcpy(h_reduce, d_reduce, blasGrid.x*sizeof(QudaSumFloat3), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();
  
  double3 gpu_result;
  gpu_result.x = 0;
  gpu_result.y = 0;
  gpu_result.z = 0;
  for (unsigned int i = 0; i < blasGrid.x; i++) {
    gpu_result.x += h_reduce[0*blasGrid.x + i];
    gpu_result.y += h_reduce[1*blasGrid.x + i];
    gpu_result.z += h_reduce[2*blasGrid.x + i];
  }

  reduceDoubleArray(&(gpu_result.x), 3);

  return gpu_result;
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
