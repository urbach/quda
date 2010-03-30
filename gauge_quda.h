#ifndef _QUDA_GAUGE_H
#define _QUDA_GAUGE_H

#include <enum_quda.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif
    
    void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, ReconstructType reconstruct, 
			  Precision precision, int *X, double anisotropy, int blockDim);
    void freeGaugeField(FullGauge *cudaCauge);
    void loadLinkToGPU(FullGauge cudaGauge, void *cpuGauge, QudaGaugeParam* param);
    void storeLinkToCPU(void* cpuGauge, FullGauge *cudaGauge, QudaGaugeParam* param);
    void createLinkQuda(FullGauge* cudaGauge, QudaGaugeParam* param);
    void createStapleQuda(FullStaple* cudaStaple, QudaGaugeParam* param);
    void freeStapleQuda(FullStaple* cudaStaple);
    void createMomQuda(FullMom* cudaMom, QudaGaugeParam* param);
    void freeMomQuda(FullMom *cudaMom);
    void storeMomToCPU(void* mom, FullMom cudaMom, QudaGaugeParam* param);
    void loadMomToGPU(FullMom cudaMom, void* mom, QudaGaugeParam* param);

#define freeLinkQuda freeGaugeField

#ifdef __cplusplus
}
#endif

#endif // _QUDA_GAUGE_H
