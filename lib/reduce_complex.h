#if (REDUCE_TYPE == REDUCE_KAHAN)

#define SH_STRIDE 2
#define REG_CREATE(x, value) QudaSumFloat x##0_r = value, x##1_r = value, x##0_i = value, x##1_i = value
#define REDUCE(x, i) dsadd(x##0_r, x##1_r, x##0_r, x##1_r, REDUCE_REAL_OPERATION(i), 0); \
  dsadd(x##0_i, x##1_i, x##0_i, x##1_i, REDUCE_IMAG_OPERATION(i), 0)
#define SH_SUM(s, i, j) dsadd(s##_r[i], s##_r[i+1], s##_r[i], s##_r[i+1], s##_r[2*j], s##_r[2*j+1]); \
  dsadd(s##_i[i], s##_i[i+1], s##_i[i], s##_i[i+1], s##_i[2*j], s##_i[2*j+1]);
#define SH_SET(s, i, x) s##_r[i] = x##0_r, s##_r[i+1] = x##1_r, s##_i[i] = x##0_i, s##_i[i+1] = x##1_i
#define SH_EVAL(s, i) s[i] + s[i+1]

#else

#define SH_STRIDE 1
#define REG_CREATE(x, value) QudaSumFloat x##_r = value, x##_i = value
#define REDUCE(x, i) x##_r += REDUCE_REAL_OPERATION(i), x##_i += REDUCE_IMAG_OPERATION(i)
#define SH_SUM(s, i, j) s##_r[i] += s##_r[j], s##_i[i] += s##_i[j]
#define SH_SET(s, i, x) s##_r[i] = x##_r, s##_i[i] = x##_i
#define SH_EVAL(s, i) s[i]

#endif

#define AUXILIARY(i) REDUCE_REAL_AUXILIARY(i); REDUCE_IMAG_AUXILIARY(i)
#define SUMFLOAT_P(x, y) QudaSumFloat *x##_r = y, *x##_i = y + SH_STRIDE*reduce_threads
#define SUMFLOAT_EQ_SUMFLOAT(a, b) QudaSumFloat a##_r = b##_r, a##_i = b##_i
#define WRITE_GLOBAL(array, i, s, j) array[i] = SH_EVAL(s##_r, j), array[i+gridDim.x] = SH_EVAL(s##_i, j)

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumFloat *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(reduce_threads) + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;
  
  REG_CREATE(sum, 0);
  
  while (i < n) {
    AUXILIARY(i);
    REDUCE(sum, i); // sum_r += REDUCE_REAL_OPERATION(i)
    i += gridSize;
  }
  
  extern __shared__ QudaSumFloat sdata[];
  SUMFLOAT_P(s, sdata + SH_STRIDE*tid);

  SH_SET(s, 0, sum);
  
  __syncthreads();
  
  if (reduce_threads >= 1024) { if (tid < 512) { SH_SUM(s, 0, 512); } __syncthreads(); }
  if (reduce_threads >= 512) { if (tid < 256) { SH_SUM(s, 0, 256); } __syncthreads(); }    
  if (reduce_threads >= 256) { if (tid < 128) { SH_SUM(s, 0, 128); } __syncthreads(); }
  if (reduce_threads >= 128) { if (tid <  64) { SH_SUM(s, 0, 64); } __syncthreads(); }    

#ifndef __DEVICE_EMULATION__
  if (tid < 32) 
#endif
    {
      volatile SUMFLOAT_EQ_SUMFLOAT(*sv, s);
      if (reduce_threads >=  64) { SH_SUM(sv, 0, 32); EMUSYNC; }
      if (reduce_threads >=  32) { SH_SUM(sv, 0, 16); EMUSYNC; }
      if (reduce_threads >=  16) { SH_SUM(sv, 0, 8); EMUSYNC; }
      if (reduce_threads >=   8) { SH_SUM(sv, 0, 4); EMUSYNC; }
      if (reduce_threads >=   4) { SH_SUM(sv, 0, 2); EMUSYNC; }
      if (reduce_threads >=   2) { SH_SUM(sv, 0, 1); EMUSYNC; }
    }
  
  // write result for this block to global mem
  if (tid == 0) { WRITE_GLOBAL(g_odata, blockIdx.x, s, 0); }
}

template <typename Float, typename Float2>
cuDoubleComplex REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {

  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_complex: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
#if (REDUCE_TYPE == REDUCE_KAHAN)
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 4 * sizeof(QudaSumComplex) : blasBlock.x * 2 * sizeof(QudaSumComplex);
#else
  size_t smemSize = (blasBlock.x <= 32) ? blasBlock.x * 2 * sizeof(QudaSumComplex) : blasBlock.x * sizeof(QudaSumComplex);
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
  cudaMemcpy(h_reduce, d_reduce, blasGrid.x*sizeof(QudaSumComplex), cudaMemcpyDeviceToHost);

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();
  
  cuDoubleComplex gpu_result;
  gpu_result.x = 0;
  gpu_result.y = 0;
  for (unsigned int i = 0; i < blasGrid.x; i++) {
    gpu_result.x += h_reduce[i];
    gpu_result.y += h_reduce[i + blasGrid.x];
  }

  reduceDoubleArray(&(gpu_result.x), 2);

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
