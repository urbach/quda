// -*- c++ -*-

//
// QMEM -- support
//
// This installs a thin layer of a CUDA backend that QUDA
// silently uses in case of configuring the package with
// QMEM support. It installs C macros for device memory
// allocation and deallocation routines and redirects those 
// calls to the device memory pool manager which is part of 
// QDP++ where previously cached lattice objects are spilled
// automatically as needed.
//
// This enables QUDA and QDP++ being able to share the same
// device memory pool and thus avoids the need to temporarily
// suspend QDP++ using GPUs. As a sideeffect this speeds up
// QUDA residual calculation and solution reconstruction as
// these parts are implemented using QDP++.
//

#ifndef QUDA_MEM
#define QUDA_MEM

#ifdef USE_QMEM

#warning "Using QMEM wrappers"
#include <qdp_mem.h>
#define cudaMalloc(dst, size) QDP_allocate(dst, size , __FILE__ , __LINE__ )
#define cudaFree(dst) QDP_free(dst)

inline cudaError_t QDP_allocate(void **dst, size_t size, char * cstrFile , int intLine )
{
  if (!QDP::Allocator::theQDPDeviceAllocator::Instance().allocate_CACHE_spilling( dst , size , cstrFile , intLine ))
    return cudaErrorMemoryAllocation;
  else
    return cudaSuccess;
}

inline void QDP_free(void *dst) 
{
  QDP::Allocator::theQDPDeviceAllocator::Instance().free( dst );
}

#endif // USE_QMEM
#endif // QUDA_MEM
