#ifndef _HISQ_FORCE_UTILS_H
#define _HISQ_FORCE_UTILS_H

#include <quda_internal.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  // The following routines are used to test the force calculation from hisq smearing
  namespace hisq{
    namespace fermion_force{

      void loadOprodToGPU(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod, int vol);
      void allocateOprodFields(void **cudaOprodEven, void **cudaOprodOdd, int vol);
      void fetchOprodFromGPU(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod, int vol);

       FullOprod createOprodQuda(int *X, QudaPrecision precision);
       void copyOprodToGPU(FullOprod cudaOprod, void *oprod, int half_volume);


    } // namespace fermion_force
  } // namespace hisq
  
#ifdef __cplusplus
}
#endif

#endif // _HISQ_FORCE_UTILS_H
