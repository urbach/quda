#ifndef _WILSON_DSLASH_REFERENCE_H
#define _WILSON_DSLASH_REFERENCE_H

#include <enum_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern int Z[4];
  extern int Vh;
  extern int V;

  void setDims(int *);

  void dslash(void *res, void **gauge, void *spinorField, double kappa,
	      double mu, QudaTwistFlavorType flavor, int oddBit,
	      int daggerBit, QudaPrecision sPrecision,
	      QudaPrecision gPrecision);
  
  void mat(void *out, void **gauge, void *in, double kappa, double mu,
	   QudaTwistFlavorType flavor, int daggerBit,
	   QudaPrecision sPrecision, QudaPrecision gPrecision);

  void matpc(void *out, void **gauge, void *in, double kappa, double mu,
	     QudaTwistFlavorType flavor, QudaMatPCType matpc_type,  
	     int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision);
	     
//BEGIN NEW
  void ndeg_twist_gamma5
  (void *out1, void *out2, void *in1, void *in2, const int dagger, const double kappa, const double mu, const double epsilon, const int V, QudaTwistGamma5Type twist);

  void ndeg_dslash
  (void *res1, void *res2, void **gaugeFull, void *spinorField1, void *spinorField2, double kappa, double mu, double epsilon, int oddBit, int daggerBit,  QudaPrecision sPrecision, QudaPrecision gPrecision);
  
  void ndeg_matpc
  (void *outEven1, void *outEven2, void **gauge, void *inEven1, void *inEven2, double kappa, double mu, double epsilon, QudaMatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision);
  
  void ndeg_mat
  (void *out1, void* out2, void **gauge, void *in1, void *in2,  double kappa, double mu, double epsilon, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision);	   
//END NEW

#ifdef __cplusplus
}
#endif

#endif // _DSLASH_REFERENCE_H
