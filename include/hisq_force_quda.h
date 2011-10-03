#ifndef _HISQ_FORCE_QUDA_H
#define _HISQ_FORCE_QUDA_H

#include "hisq_force_utils.h"

#ifdef __cplusplus
extern "C"{
#endif

namespace hisq {
  namespace fermion_force {
  void hisq_force_init_cuda(QudaGaugeParam* param);
  void hisq_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff, FullOprod cudaOprod, FullGauge cudaSiteLink, FullMom cudaMom, FullGauge cudaMomMatrix, QudaGaugeParam* param);
 } // namespace fermion_force
}  // namespace hisq

#ifdef __cplusplus
}
#endif

#endif // _HISQ_FORCE_QUDA_H
