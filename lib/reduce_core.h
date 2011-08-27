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
