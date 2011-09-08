#include <tune_quda.h>

void TuneBase::Benchmark(dim3 &block)  {

  int count = 10;
  int threadBlockMin = 32;
  int threadBlockMax = 256;
  double time;
  double timeMin = 1e10;
  double gflopsMax = 0.0;
  dim3 blockOpt(1,1,1);

  cudaError_t error;

  for (int threads=threadBlockMin; threads<=threadBlockMax; threads+=32) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    block = dim3(threads,1,1);

    Flops(); // resets the flops counter

    cudaGetLastError(); // clear error counter

    for (int c=0; c<count; c++) Apply();

    error = cudaGetLastError();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float runTime;
    cudaEventElapsedTime(&runTime, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    time = runTime / 1000;
    double flops = (double)Flops();
    double gflops = (flops*1e-9)/(time);

    if (time < timeMin && error == cudaSuccess) {
      timeMin = time;
      blockOpt = block;
      gflopsMax = gflops;
    }

    if (verbose >= QUDA_DEBUG_VERBOSE && error == cudaSuccess) 
      printfQuda("%-15s %d %f s, flops = %e, Gflop/s = %f\n", name, threads, time, (double)flops, gflops);
  }

  block = blockOpt;
  Flops(); // reset the flop counter

  if (block.x == 1) {
    printfQuda("Auto-tuning failed for %s\n", name);
  }

  if (verbose >= QUDA_VERBOSE) 
    printfQuda("Tuned %-15s with (%d,%d,%d) threads per block, Gflop/s = %f\n", name, block.x, block.y, block.z, gflopsMax);    

}
void TuneBase::BenchmarkMulti(dim3* block, int n)  
{
  int count = 50;
  int threadBlockMin = 32;
  int threadBlockMax = 256;
  double time;
  double timeMin = 1e10;
  double gflopsMax = 0.0;
  dim3 blockOpt(1,1,1);
  
  cudaError_t error;
  for(int func_idx = 0;func_idx < n; func_idx++){
    for (int threads=threadBlockMin; threads<=threadBlockMax; threads+=32) {
      cudaEvent_t start, end;
      cudaEventCreate(&start);
      cudaEventCreate(&end);
      cudaEventRecord(start, 0);
      cudaEventSynchronize(start);
      
      block[func_idx] = dim3(threads,1,1);
      
      Flops(); // resets the flops counter
      
      cudaGetLastError(); // clear error counter
      
      for (int c=0; c<count; c++) ApplyMulti(func_idx);
      
      error = cudaGetLastError();
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      float runTime;
      cudaEventElapsedTime(&runTime, start, end);
      cudaEventDestroy(start);
      cudaEventDestroy(end);
      
      time = runTime / 1000;
      double flops = (double)Flops();
      double gflops = (flops*1e-9)/(time);
      
      if (time < timeMin && error == cudaSuccess) {
	timeMin = time;
	blockOpt = block[func_idx];
	gflopsMax = gflops;
      }
      
      if (verbose >= QUDA_DEBUG_VERBOSE && error == cudaSuccess) 
	printfQuda("%-15s %d %f s, flops = %e, Gflop/s = %f\n", name, threads, time, (double)flops, gflops);
    }
    
    block[func_idx] = blockOpt;
    Flops(); // reset the flop counter
    
    if (block[func_idx].x == 1) {
      printfQuda("Auto-tuning failed for %s\n", name);
    }
    
    if (verbose >= QUDA_VERBOSE){ 
      printfQuda("Tuned %-15s with (%d,%d,%d) threads per block, Gflop/s = %f\n", name, 
		 block[func_idx].x, block[func_idx].y, block[func_idx].z, gflopsMax);    
    }
  }
}
