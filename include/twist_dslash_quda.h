#ifndef _TWIST_DSLASH_QUDA_H
#define _TWIST_DSLASH_QUDA_H

#include <quda_internal.h>
#include <dslash_quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  extern unsigned long long dslash_quda_flops;
  extern unsigned long long dslash_quda_bytes;

  //routines defined in dslash_quda.cu
  //void initCache(void);
  //int dslashCudaSharedBytes(Precision spinor_prec, int blockDim);

 
  /*---------------------routines for the twisted mass stuff--------------------------*/
  
  // Double precision twisted mass dslash operator
  void twistDslashDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit, double kappa, double mu);
		   
  void twistDslashXpayDCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double kappa, double mu);		   

  // Single precision  twisted mass dslash operator
  void twistDslashSCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit, float kappa, float mu);
		   
  void twistDslashXpaySCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, float kappa, float mu);		   


  // Half precision twisted mass dslash operator
  void twistDslashHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit, float kappa, float mu);
		   
  void twistDslashXpayHCuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, float kappa, float mu);		   


  // wrapper to above
  void twistDslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
		  int parity, int dagger, double kappa, double mu);
		  
  void twistDslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
		      int parity, int dagger, ParitySpinor x, double kappa, double mu);
		      
  void twistMatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		 double kappa, double mu, ParitySpinor tmp, MatPCType matpc_type,
		 int daggerBit);
		 
  void twistMatPCDagMatPCCuda(ParitySpinor outEven, FullGauge gauge,
			 ParitySpinor inEven, double kappa, double mu, ParitySpinor tmp,
			 MatPCType matpc_type);	
			 
  void twistGamma5Cuda(ParitySpinor spinor, double a, double b);			 
		  
		  

#ifdef __cplusplus
}
#endif

#endif // _TWIST_DLASH_QUDA_H
