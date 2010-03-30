#ifndef _QUDA_DSLASH_H
#define _QUDA_DSLASH_H

#include <cuComplex.h>

#include "quda.h"

#define gaugeSiteSize 18 // real numbers per link
#define spinorSiteSize 6 // real numbers per spinor
#define cloverSiteSize 72 // real numbers per block-diagonal clover matrix
#define momSiteSize 10 //real numbers for compressed momentum 
#define hwSiteSize 12 //real numbers for half wilson vector

#ifdef __cplusplus
extern "C" {
#endif

    extern FullGauge cudaGaugePrecise;
    extern FullGauge cudaGaugeSloppy;
    extern FullGauge cudaFatLinkPrecise;
    extern FullGauge cudaFatLinkSloppy;
    extern FullGauge cudaLongLinkPrecise;    
    extern FullGauge cudaLongLinkSloppy;

    extern QudaGaugeParam *gauge_param;
    extern QudaInvertParam *invert_param;

    extern FullClover cudaClover;

    // ---------- dslash_quda.cu ----------

    int dslashCudaSharedBytes(Precision spinor_prec, int blockDim);
    void initDslashCuda(FullGauge);
    void bindGaugeTex(FullGauge gauge, int oddBit);
    void bindFatLongLinkTex(FullGauge flink, FullGauge llink, int oddBit);

    // Double precision routines
    void dslashDCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor,
			int oddBit, int daggerBit);
    void dslashXpayDCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
			 int oddBit, int daggerBit, ParitySpinor x, double a);
    
    // Single precision routines
    void dslashSCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor,
		     int oddBit, int daggerBit);
    void dslashXpaySCuda(ParitySpinor res, FullGauge link, FullGauge longlink, ParitySpinor spinor, 
			 int oddBit, int daggerBit, ParitySpinor x, double a);
    
    // Half precision dslash routines
    void dslashHCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor,
		     int oddBit, int daggerBit);
    void dslashXpayHCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
			 int oddBit, int daggerBit, ParitySpinor x, double a);

    // wrapper to above
    void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger);
    void dslashCuda_st(ParitySpinor out, FullGauge flink, FullGauge llink, ParitySpinor in, int parity, int dagger);
    void dslashFullCuda(FullSpinor out, FullGauge cudaFatLink, FullGauge cudaLongLink, FullSpinor in, int dagger);
    void dslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
			ParitySpinor x, double a);
    void dslashXpayCuda_st(ParitySpinor out, FullGauge fatlink, FullGauge longlink, ParitySpinor in, int parity, int dagger,
			   ParitySpinor x, double a);
    void dslashAxpyCuda(ParitySpinor out, FullGauge fatlink, FullGauge longlink, 
			ParitySpinor in,  int parity, int dagger,
			ParitySpinor x, double a) ;   
    void dslashAxpyFullCuda(FullSpinor out, FullGauge fatlink, FullGauge longlink, 
			    FullSpinor in,  int dagger,
			    FullSpinor x, double a) ;   
    // Full Wilson matrix
    void MatCuda_st(FullSpinor out, FullGauge fatlink, FullGauge longlink, FullSpinor in, double kappa, int daggerBit);
    
    void MatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven, 
		   double kappa, ParitySpinor tmp, MatPCType matpc_type, int daggerBit);
    void MatPCCuda_st(ParitySpinor outEven, FullGauge fatlink, FullGauge longlink, ParitySpinor inEven, 
		      double kappa, ParitySpinor tmp, MatPCType matpc_type, int daggerBit);

    void MatPCDagMatPCCuda(ParitySpinor outEven, FullGauge gauge, ParitySpinor inEven,
			   double kappa, ParitySpinor tmp, MatPCType matpc_type);
    void MatPCDagMatPCCuda_st(ParitySpinor outEven, FullGauge fatlink, FullGauge longlink, ParitySpinor inEven,
			      double kappa, ParitySpinor tmp, MatPCType matpc_type);
  
    void MatDagMatCuda(ParitySpinor out, FullGauge fatlink, FullGauge longlink, ParitySpinor in, 
		       double kappa, ParitySpinor tmp, int);
	
    // -- inv_cg_cuda.cpp
    int invertCgCuda(ParitySpinor x, ParitySpinor b, FullGauge fatlink, FullGauge longlink, 
		      FullGauge fatlinkSloppy, FullGauge longlinkSloppy, ParitySpinor tmp, QudaInvertParam *param);        
    int invertCgCuda_milc_parity(ParitySpinor x, ParitySpinor source, FullGauge fatlinkPrecise, FullGauge longlinkPrecise,
				  FullGauge fatlinkSloppy, FullGauge longlinkSloppy, ParitySpinor tmp, QudaInvertParam *perf, double mass, int oddBit);
    int invertCgCuda_milc_full(FullSpinor x, FullSpinor source, FullGauge fatlinkPrecise, FullGauge longlinkPrecise,
				FullGauge fatlinkSloppy, FullGauge longlinkSloppy, FullSpinor tmp, QudaInvertParam *perf, double mass);
    int invertCgCuda_milc_multi_mass_parity(ParitySpinor* x, ParitySpinor source, FullGauge fatlinkPrecise, FullGauge longlinkPrecise,
					    FullGauge fatlinkSloppy, FullGauge longlinkSloppy, ParitySpinor tmp, QudaInvertParam *perf,
					    double* offsets, int num_offsets, int oddBit, double* residue_sq);
    
    
#ifdef __cplusplus
}
#endif

#endif // _QUDA_DLASH_H
