#ifndef _UNITARIZE_QUDA_H
#define _UNITARIZE_QUDA_H


#include "quda.h"

#ifdef __cplusplus
extern "C"{
#endif

void unitarize_cuda_hc(FullGauge cudaOutLink, FullGauge cudaInLink,
    const QudaGaugeParam* const param);

void unitarize_cuda_si(FullGauge cudaOutLink, FullGauge cudaInLink,
    const QudaGaugeParam* const param,
    int num_its);

void unitarize_init_cuda(QudaGaugeParam* param);


#ifdef __cplusplus
}
#endif


#endif  // _UNITARIZE_QUDA_H
