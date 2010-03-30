#include <cuComplex.h>
#include <enum_quda.h>

#ifndef _QUDA_BLAS_H
#define _QUDA_BLAS_H

#ifdef __cplusplus
extern "C" {
#endif

    // ---------- blas_quda.cu ----------
  
    void zeroCuda(ParitySpinor a);
    void zeroFullCuda(FullSpinor a);
    void copyCuda(ParitySpinor dst, ParitySpinor src);
    void copyFullCuda(FullSpinor dst, FullSpinor src);  
    double axpyNormCuda(double a, ParitySpinor x, ParitySpinor y);
    double axpyNormFullCuda(double a, FullSpinor x, FullSpinor y);
    double normCuda(ParitySpinor b);
    double relativeNormCuda(ParitySpinor p, ParitySpinor q);
    double normFullCuda(FullSpinor a);
    double reDotProductCuda(ParitySpinor a, ParitySpinor b);
    double reDotProductFullCuda(FullSpinor a, FullSpinor b);
    double xmyNormCuda(ParitySpinor a, ParitySpinor b);
    double xmyNormFullCuda(FullSpinor a, FullSpinor b);

    void axpbyCuda(double a, ParitySpinor x, double b, ParitySpinor y);
    void axpyCuda(double a, ParitySpinor x, ParitySpinor y);
    void axpyFullCuda(double a, FullSpinor x, FullSpinor y);
    void axCuda(double a, ParitySpinor x);
    void xpyCuda(ParitySpinor x, ParitySpinor y);
    void xpyFullCuda(FullSpinor x, FullSpinor y);
    void xpayCuda(ParitySpinor x, double a, ParitySpinor y);
    void xpayFullCuda(FullSpinor x, double a, FullSpinor y);
    void mxpyCuda(ParitySpinor x, ParitySpinor y);
    void mxpyFullCuda(FullSpinor x, FullSpinor y);
  
    void axpyZpbxCuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b);
    void axpyZpbxFullCuda(double a, FullSpinor x, FullSpinor y, FullSpinor z, double b);
    void axpyBzpcxCuda(double a, ParitySpinor x, ParitySpinor y, double b, ParitySpinor z, double c);

    void blasTest();
    void axpbyTest();
    
#ifdef __cplusplus
}
#endif

#endif // _QUDA_BLAS_H
