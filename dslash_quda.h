// Ver. 09.12.a

#ifndef DSLASH_DWF_QUDA_H
#define DSLASH_DWF_QUDA_H

#include <cuComplex.h>

#include <quda.h>


#define gaugeSiteSize 18 // real numbers per link
#define spinorSiteSize 24 // real numbers per spinor

// Got rid of clover.

#define BLOCK_DIM (64) // threads per block
#define GRID_DIM (Nh_5d/BLOCK_DIM) // there are Nh_5d threads in total.

//J  Gauge doesn't care if it's dwf.  Use Nh_4d.
#define PACKED12_GAUGE_BYTES (4*Nh_4d*12*sizeof(float))
#define PACKED8_GAUGE_BYTES (4*Nh_4d*8*sizeof(float))

// Got rid of clover.

#ifdef __cplusplus
extern "C" {
#endif

  extern FullGauge cudaGaugePrecise;
  extern FullGauge cudaGaugeSloppy;

  extern QudaGaugeParam *gauge_param;
  extern QudaInvertParam *invert_param;


// ---------- dslash_quda.cu ----------

  int dslashCudaSharedBytes();
  void setCudaGaugeParam();
  void bindGaugeTex(FullGauge gauge, int oddBit);

  // Double precision routines
  void dslashD_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit, double mferm);
  //ok
  void dslashXpayD_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double mferm, double a);

  // Single precision routines
  void dslashS_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit, double mferm);
  //ok
  void dslashXpayS_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double mferm, double a);

  // Half precision dslash routines
  void dslashH_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor,
		   int oddBit, int daggerBit, double mferm);
  void dslashXpayH_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		       int oddBit, int daggerBit, ParitySpinor x, double mferm, double a);

  // wrapper to above
  void dslash_dwf_Cuda(ParitySpinor out, FullGauge gauge, ParitySpinor in,
    int parity, int dagger, double mferm);
  //ok
  void dslashXpay_dwf_Cuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
		      ParitySpinor x, double mferm, double a);

  // Full DWF matrix.  See dslash_quda.cu.
  // ok
  void Mat_dwf_Cuda(FullSpinor out, FullGauge gauge, FullSpinor in, 
                  double kappa, int daggerBit, double mferm);

  //ok
  void MatPC_dwf_Cuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		 double kappa, ParitySpinor tmp, MatPCType matpc_type, int daggerBit,
     double mferm);

  //ok
  void MatPCDagMatPC_dwf_Cuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven,
			 double kappa, ParitySpinor tmp, MatPCType matpc_type, double mferm);
  
  
  // -- inv_cg_quda.cpp
  void invertCgCuda(ParitySpinor x, ParitySpinor b, FullGauge gauge, 
		    ParitySpinor tmp, QudaInvertParam *param);
  
  // -- inv_bicgstab_quda.cpp
  void invertBiCGstabCuda(ParitySpinor x, ParitySpinor b, FullGauge gaugeSloppy, 
			  FullGauge gaugePrecise, ParitySpinor tmp, 
			  QudaInvertParam *param, DagType dag_type);
  
#ifdef __cplusplus
}
#endif

#endif

