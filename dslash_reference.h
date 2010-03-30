#include <blas_reference.h>

#ifndef _QUDA_DSLASH_REF_H
#define _QUDA_DSLASH_REF_H

#ifdef __cplusplus
extern "C" {
#endif
    
    extern int Z[4];
    extern int Vh;
    extern int V;
    
    void setDims(int *);
    
    void dslash(void *res, void ** fatlink, void** longlink, void *spinorField,
		int oddBit, int daggerBit, Precision sPrecision, Precision gPrecision);
    
    void mat(void *out, void **fatlink, void** longlink, void *in, double kappa, int daggerBit,
	     Precision sPrecision, Precision gPrecision);
    
    void matpc(void *out, void **fatlink, void ** longlink, void *in, double kappa, MatPCType matpc_type, 
	       int daggerBit, Precision sPrecision, Precision gPrecision);
    void matdagmat_milc(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
			Precision sPrecision, Precision gPrecision, void* tmp, QudaParity parity);    
    void mymatdagmat_milc(void *out, void **fatlink, void** longlink, void *in, double mass, int dagger_bit,
			  Precision sPrecision, Precision gPrecision, void* tmp, QudaParity parity);    
#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_REF_H
