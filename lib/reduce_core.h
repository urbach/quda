#define AUXILIARY(i) REDUCE_AUXILIARY(i);
#define SUMFLOAT_P(x, y) QudaSumFloat *s = y;

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

#define WRITE_GLOBAL(x, i, s, j) x[i] = SH_EVAL(s,j)

__global__ void REDUCE_FUNC_NAME(Kernel) (REDUCE_TYPES, QudaSumFloat *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(reduce_threads) + threadIdx.x;
  unsigned int gridSize = reduce_threads*gridDim.x;
  
  REG_CREATE(sum, 0); // QudaSumFloat sum = 0;
  
  while (i < n) {
    AUXILIARY(i);
    REDUCE(sum, i); // sum += REDUCE_OPERATION(i);
    i += gridSize;
  }
  
  extern __shared__ QudaSumFloat sdata[];
  SUMFLOAT_P(s, sdata + SH_STRIDE*tid);

  SH_SET(s, 0, sum); // s[0] = sum;

  __syncthreads();
  
  // do reduction in shared mem
  if (reduce_threads>=1024){ if (tid<512) { SH_SUM(s, 0, 512); } __syncthreads(); }
  if (reduce_threads>=512) { if (tid<256) { SH_SUM(s, 0, 256); } __syncthreads(); }
  if (reduce_threads>=256) { if (tid<128) { SH_SUM(s, 0, 128); } __syncthreads(); }
  if (reduce_threads>=128) { if (tid<64) { SH_SUM(s, 0, 64); } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
  if (tid < 32) 
#endif
    {
      volatile QudaSumFloat *sv = s;
      if (reduce_threads >=  64) { SH_SUM(sv, 0, 32); EMUSYNC; }
      if (reduce_threads >=  32) { SH_SUM(sv, 0, 16); EMUSYNC; }
      if (reduce_threads >=  16) { SH_SUM(sv, 0, 8); EMUSYNC; }
      if (reduce_threads >=   8) { SH_SUM(sv, 0, 4); EMUSYNC; }
      if (reduce_threads >=   4) { SH_SUM(sv, 0, 2); EMUSYNC; }
      if (reduce_threads >=   2) { SH_SUM(sv, 0, 1); EMUSYNC; }
    }
  
  // write result for this block to global mem as single float
  if (tid == 0) { WRITE_GLOBAL(g_odata, blockIdx.x, s, 0); }
}

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
