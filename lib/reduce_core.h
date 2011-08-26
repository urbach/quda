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
  SUMFLOAT_P(s, sdata + tid);

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
      volatile SUMFLOAT_EQ_SUMFLOAT(*sv, s);
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
