#include "reduce_core.h"

template <typename R, typename FloatN>
R REDUCE_FUNC_NAME(Cuda) (REDUCE_TYPES, int n, int kernel, QudaPrecision precision) {
  setBlock(kernel, n, precision);
  
  if (blasGrid.x > REDUCE_MAX_BLOCKS) {
    errorQuda("reduce_core: grid size %d must be smaller than %d", blasGrid.x, REDUCE_MAX_BLOCKS);
  }
  
  const int N = sizeof(R) / sizeof(double); // how many elements are we reducing to

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
  
#ifdef DEVICE_REDUCTION
  {
#define ZERO_COPY_INIT 0x7FEFFFFFFFFFFFFF  // maximum double precision value
    for (unsigned int j=0; j<N; j++) h_reduce[j] = ZERO_COPY_INIT;

    // do final device-side reduction and the write the result to zero-copy memory on the host
    dim3 block(blasGrid.x, 1, 1);
    dim3 grid(N, 1); // each thread block handles the different components
    size_t smemSize = block.x * sizeof(QudaSumFloat) * (block.x <= 32 ? 2 : 1);
    if (block.x == 1) {
      sumDKernel<1><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 2) {
      sumDKernel<2><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 4) {
      sumDKernel<4><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 8) {
      sumDKernel<8><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 16) {
      sumDKernel<16><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 32) {
      sumDKernel<32><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 64) {
      sumDKernel<64><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 128) {
      sumDKernel<128><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 256) {
      sumDKernel<256><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 512) {
      sumDKernel<512><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else if (block.x == 1024) {
      sumDKernel<1024><<<grid, block, smemSize>>>(d_reduce, hd_reduce, N*blasGrid.x);
    } else {
      errorQuda("Final reduction not implemented for %d threads", block.x);
    }

  }

  // Need to synchronize since zero-copy write is asynchronous
  //cudaThreadSynchronize();
  R sum_h;
  double *sum_p = (double*)&sum_h; // cast return type as an array of doubles
  volatile double *h_reduce_v = h_reduce;
  for (unsigned int j=0; j<N; j++) {
    while (h_reduce_v[j] == ZERO_COPY_INIT) {  } // fastest synchronize is to poll on the cpu until kernel completes
    sum_p[j] = h_reduce_v[j];
  }
#else
  // copy result from device to host, and perform final reduction on CPU
  cudaMemcpy(h_reduce, d_reduce, N*blasGrid.x*sizeof(QudaSumFloat), cudaMemcpyDeviceToHost);

  R sum_h;
  double *sum_p = (double*)&sum_h; // cast return type as an array of doubles
  for (unsigned int j=0; j<N; j++) {
    sum_p[j] = 0.0;
    for (unsigned int i = 0; i<blasGrid.x; i++) sum_p[j] += h_reduce[j*blasGrid.x + i];
  }
#endif

  // for a tuning run, let blas_test check the error condition
  if (!blasTuning) checkCudaError();

  reduceDoubleArray(sum_p, N);

  return sum_h;
}
