#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "dslash_quda.h"


static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

// Double precision gauge field
texture<int4, 1> fatLink0TexDouble;
texture<int4, 1> fatLink1TexDouble;

texture<int4, 1> longLink0TexDouble;
texture<int4, 1> longLink1TexDouble;



// Single precision gauge field
texture<float2, 1, cudaReadModeElementType> fatLink0TexSingle;
texture<float2, 1, cudaReadModeElementType> fatLink1TexSingle;

texture<float4, 1, cudaReadModeElementType> longLink0TexSingle;
texture<float4, 1, cudaReadModeElementType> longLink1TexSingle;

texture<float2, 1, cudaReadModeElementType> longLink0TexSingle_norecon;
texture<float2, 1, cudaReadModeElementType> longLink1TexSingle_norecon;

// Half precision gauge field
texture<short2, 1, cudaReadModeNormalizedFloat> fatLink0TexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> fatLink1TexHalf;

texture<short4, 1, cudaReadModeNormalizedFloat> longLink0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> longLink1TexHalf;

//Single precision for site link
texture<float4, 1, cudaReadModeElementType> siteLink0TexSingle;
texture<float4, 1, cudaReadModeElementType> siteLink1TexSingle;

// Single precision mulink field
texture<float2, 1, cudaReadModeElementType> muLink0TexSingle;
texture<float2, 1, cudaReadModeElementType> muLink1TexSingle;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float2, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short2, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<float, 1, cudaReadModeElementType> spinorTexNorm;

// Double precision accumulate spinor field
texture<int4, 1> accumTexDouble;

// Single precision accumulate spinor field
texture<float2, 1, cudaReadModeElementType> accumTexSingle;

// Half precision accumulate spinor field
texture<short2, 1, cudaReadModeNormalizedFloat> accumTexHalf;
texture<float, 1, cudaReadModeElementType> accumTexNorm;

QudaGaugeParam *gauge_param;
QudaInvertParam *invert_param;

__constant__ int gauge_fixed;

// single precision constants
__constant__ float anisotropy_f;
__constant__ float coeff_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double coeff;
__constant__ double t_boundary;

static int initDslash = 0;

#include "kernel_common.cu"
#include "dslash_def.h"

void 
initDslashCuda(FullGauge gauge) 
{
    if (initDslash){
	return;
    }
    init_kernel_cuda(gauge_param);

    int Vh = gauge.volume;
    
    if (gauge.blockDim%64 != 0) {
	printf("Sorry, block size not set approriately\n");
	exit(-1);
    }
    
    if (Vh%gauge.blockDim !=0) {
	printf("Sorry, volume is not a multiple of number of threads %d\n", gauge.blockDim);
	exit(-1);
    }
    
 
    int gf = (gauge_param->gauge_fix == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
    cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));
    
    cudaMemcpyToSymbol("anisotropy", &(gauge_param->anisotropy), sizeof(double));
    
    double t_bc = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
    cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));
    
    float anisotropy_f = gauge_param->anisotropy;
    cudaMemcpyToSymbol("anisotropy_f", &(anisotropy_f), sizeof(float));
    
    float coeff_f = -24.0*gauge_param->anisotropy*gauge_param->anisotropy;
    cudaMemcpyToSymbol("coeff_f", &(coeff_f), sizeof(float));
    
    double coeff = -24.0*gauge_param->anisotropy*gauge_param->anisotropy;
    cudaMemcpyToSymbol("coeff", &(coeff), sizeof(double));
    
    float t_bc_f = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
    cudaMemcpyToSymbol("t_boundary_f", &(t_bc_f), sizeof(float));

    float h_pi_f = M_PI;
    cudaMemcpyToSymbol("pi_f", &(h_pi_f), sizeof(float));
    
    initDslash = 1;
}

void 
bindFatLongLinkTex(FullGauge flink, FullGauge llink, int oddBit) 
{
    if (flink.precision == QUDA_DOUBLE_PRECISION){
	if (oddBit) {
	    cudaBindTexture(0, fatLink0TexDouble, flink.odd, flink.bytes); 
	    cudaBindTexture(0, fatLink1TexDouble, flink.even, flink.bytes);
	    cudaBindTexture(0, longLink0TexDouble, llink.odd, llink.bytes); 
	    cudaBindTexture(0, longLink1TexDouble, llink.even, llink.bytes);
	    
	} else {
	    cudaBindTexture(0, fatLink0TexDouble, flink.even, flink.bytes);
	    cudaBindTexture(0, fatLink1TexDouble, flink.odd, flink.bytes); 
	    cudaBindTexture(0, longLink0TexDouble, llink.even, llink.bytes);
	    cudaBindTexture(0, longLink1TexDouble, llink.odd, llink.bytes); 
	}	
    }else if (flink.precision == QUDA_SINGLE_PRECISION) {

	if (oddBit) {
	    cudaBindTexture(0, fatLink0TexSingle, flink.odd, flink.bytes); 
	    cudaBindTexture(0, fatLink1TexSingle, flink.even, flink.bytes);
	    
	} else {
	    cudaBindTexture(0, fatLink0TexSingle, flink.even, flink.bytes);
	    cudaBindTexture(0, fatLink1TexSingle, flink.odd, flink.bytes); 
	}
	
	
	if (llink.reconstruct == QUDA_RECONSTRUCT_NO){ //18 reconstruct 
	    if (oddBit) {
		cudaBindTexture(0, longLink0TexSingle_norecon, llink.odd, llink.bytes); 
		cudaBindTexture(0, longLink1TexSingle_norecon, llink.even, llink.bytes);
		
	    } else {
		cudaBindTexture(0, longLink0TexSingle_norecon, llink.even, llink.bytes);
		cudaBindTexture(0, longLink1TexSingle_norecon, llink.odd, llink.bytes); 
	    }
	}else{
	    if (oddBit) {
		cudaBindTexture(0, longLink0TexSingle, llink.odd, llink.bytes); 
		cudaBindTexture(0, longLink1TexSingle, llink.even, llink.bytes);
		
	    } else {
		cudaBindTexture(0, longLink0TexSingle, llink.even, llink.bytes);
		cudaBindTexture(0, longLink1TexSingle, llink.odd, llink.bytes); 
	    }

	}
    } else {
	if (oddBit) {
	    cudaBindTexture(0, fatLink0TexHalf, flink.odd, flink.bytes); 
	    cudaBindTexture(0, fatLink1TexHalf, flink.even, flink.bytes);
	    cudaBindTexture(0, longLink0TexHalf, llink.odd, llink.bytes); 
	    cudaBindTexture(0, longLink1TexHalf, llink.even, llink.bytes);
	    
	} else {
	    cudaBindTexture(0, fatLink0TexHalf, flink.even, flink.bytes);
	    cudaBindTexture(0, fatLink1TexHalf, flink.odd, flink.bytes); 
	    cudaBindTexture(0, longLink0TexHalf, llink.even, llink.bytes);
	    cudaBindTexture(0, longLink1TexHalf, llink.odd, llink.bytes); 
	}
	
    }
}
// ----------------------------------------------------------------------

#define checkSpinor(out, in) do{					\
	if (in.precision != out.precision) {				\
	    printf("Error in dslash quda: input and out spinor precision's don't match, file: %s, line %d, funciton \n", __FILE__, __LINE__, __FUNCTION__); \
	    exit(-1);							\
	}								\
    }while(0)								\
	
#define checkGaugeSpinor(spinor,gauge) \
    if (spinor.volume != gauge.volume) { \
	printf("Error, spinor volume %d doesn't match gauge volume %d, function(%s), line=%d\n", spinor.volume, gauge.volume, __FUNCTION__, __LINE__); \
	exit(-1); \
}

void temp_checkGaugeSpinor(ParitySpinor spinor, FullGauge gauge) {
  if (spinor.volume != gauge.volume) {
    printf("Error, spinor volume %d doesn't match gauge volume %d\n", spinor.volume, gauge.volume);
    exit(-1);
  }


}

void dslashCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger) {
  if (!initDslash) initDslashCuda(gauge);
  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
      //dslashDCuda(out, gauge, in, parity, dagger);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
      //dslashSCuda(out, gauge, in, parity, dagger);
  } else if (in.precision == QUDA_HALF_PRECISION) {
      //dslashHCuda(out, gauge, in, parity, dagger);
  }
}


void dslashCuda_st(ParitySpinor out, FullGauge cudaFatLink, FullGauge cudaLongLink, ParitySpinor in, int parity, int dagger) 
{
    if (!initDslash) {
	initDslashCuda(cudaFatLink);
    }
    checkSpinor(in, out);
    checkGaugeSpinor(in, cudaFatLink);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
	dslashDCuda(out, cudaFatLink, cudaLongLink, in, parity, dagger);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
	dslashSCuda(out, cudaFatLink, cudaLongLink, in, parity, dagger);
    } else if (in.precision == QUDA_HALF_PRECISION) {
	dslashHCuda(out, cudaFatLink, cudaLongLink, in, parity, dagger);
    }
}

void 
dslashFullCuda(FullSpinor out, FullGauge cudaFatLink, FullGauge cudaLongLink, FullSpinor in, int dagger) 
{
    if (!initDslash) {
	initDslashCuda(cudaFatLink);
    }
    checkSpinor(in.even, out.even);
    checkGaugeSpinor(in.even, cudaFatLink);
    
    if (in.even.precision == QUDA_DOUBLE_PRECISION) {
	dslashDCuda(out.odd, cudaFatLink, cudaLongLink, in.even, 1, dagger);
	dslashDCuda(out.even, cudaFatLink, cudaLongLink, in.odd, 0, dagger);
    } else if (in.even.precision == QUDA_SINGLE_PRECISION) {
	dslashSCuda(out.odd, cudaFatLink, cudaLongLink, in.even, 1, dagger);
	dslashSCuda(out.even, cudaFatLink, cudaLongLink, in.odd, 0, dagger);
	
    } else if (in.even.precision == QUDA_HALF_PRECISION) {
	dslashHCuda(out.odd, cudaFatLink, cudaLongLink, in.even, 1, dagger);
	dslashHCuda(out.even, cudaFatLink, cudaLongLink, in.odd, 0, dagger);
    }
}

void dslashDCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {
    
  dim3 gridDim(res.volume/llink.blockDim, 1, 1);
  dim3 blockDim(llink.blockDim, 1, 1);
  
  bindFatLongLinkTex(flink, llink, oddBit); CUERR;
  
  int spinor_bytes = res.length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 

  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (llink.precision == QUDA_DOUBLE_PRECISION) {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	  dslashDD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      } else {
	dslashDD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	if (!daggerBit) {
	    dslashDD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	} else {
	    dslashDD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	}
    }else{
	if (!daggerBit) {
	    dslashDD18Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	} else {
	    dslashDD18DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	}

    }
  } else if (llink.precision == QUDA_SINGLE_PRECISION) {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	  dslashSD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      } else {
	dslashSD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
      if (!daggerBit) {
	dslashSD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      } else {
	dslashSD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      }
    }else{
	if (!daggerBit) {
	    dslashSD18Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	} else {
	    dslashSD18DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	}

    }
  } else {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	  dslashHD12Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      } else {
	  dslashHD12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	if (!daggerBit) {
	    dslashHD8Kernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	} else {
	    dslashHD8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit); CUERR;
	}
    }else{
	printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	exit(1);
    }
  }
  
}


void 
dslashSCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
	    int oddBit, int daggerBit) 
{
    
    dim3 gridDim(res.volume/llink.blockDim, 1, 1);
    dim3 blockDim(llink.blockDim, 1, 1);
    
    bindFatLongLinkTex(flink, llink, oddBit);

    int spinor_bytes = res.length*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
    
    int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

    if (llink.precision == QUDA_DOUBLE_PRECISION) {
	if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	    if (!daggerBit) {
		dslashDS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashDS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	} else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	    if (!daggerBit) {
		dslashDS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashDS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	}else{
	    if (!daggerBit) {
		dslashDS18Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashDS18DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	}
    } else if (llink.precision == QUDA_SINGLE_PRECISION) {
	if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	    if (!daggerBit) {
		dslashSS12Kernel<<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashSS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	} else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	    if (!daggerBit) {
		dslashSS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashSS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	}else{
	    if (!daggerBit) {
		dslashSS18Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashSS18DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	}

    } else {
	if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	    if (!daggerBit) {
		dslashHS12Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashHS12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	} else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	    if (!daggerBit) {
		dslashHS8Kernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    } else {
		dslashHS8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit); CUERR;
	    }
	}else{
	    printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	    exit(1);
	}
    }
  
}


void dslashHCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {

  dim3 gridDim(res.volume/llink.blockDim, 1, 1);
  dim3 blockDim(llink.blockDim, 1, 1);

  bindFatLongLinkTex(flink, llink, oddBit);
  
  int spinor_bytes = res.length*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, 2*spinor_bytes/spinorSiteSize); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (llink.precision == QUDA_DOUBLE_PRECISION) {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	  dslashDH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
      } else {
	  dslashDH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	if (!daggerBit) {
	    dslashDH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	} else {
	    dslashDH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	}
    }else{
	printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	exit(1);
    }
  } else if (llink.precision == QUDA_SINGLE_PRECISION) {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  } else {
	      dslashSH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	  if (!daggerBit) {
	      dslashSH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  } else {
	      dslashSH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
      }
  } else {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashHH12Kernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  } else {
	      dslashHH12DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8){
	  if (!daggerBit) {
	      dslashHH8Kernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  } else {
	      dslashHH8DaggerKernel <<<gridDim, blockDim, shared_bytes>>> ((short2*)res.spinor, (float*)res.spinorNorm, oddBit); CUERR;
	  }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
      }
  }
  
}

void dslashXpayCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
		    ParitySpinor x, double a) {
  if (!initDslash) initDslashCuda(gauge);
  checkSpinor(in, out);
  checkGaugeSpinor(in, gauge);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
      //dslashXpayDCuda(out, gauge, in, parity, dagger, x, a);
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
      //dslashXpaySCuda(out, gauge, in, parity, dagger, x, a);
  } else if (in.precision == QUDA_HALF_PRECISION) {
      //dslashXpayHCuda(out, gauge, in, parity, dagger, x, a);
  }
}

void dslashXpayCuda_st(ParitySpinor out, FullGauge fatlink, FullGauge longlink, ParitySpinor in, 
		       int parity, int dagger,
		       ParitySpinor x, double a) 
{
    if (!initDslash) initDslashCuda(fatlink);
    checkSpinor(in, out);
    checkGaugeSpinor(in, fatlink);
    checkGaugeSpinor(in, longlink);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
	dslashXpayDCuda(out, fatlink, longlink, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
	dslashXpaySCuda(out, fatlink, longlink, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_HALF_PRECISION) {
	dslashXpayHCuda(out, fatlink, longlink, in, parity, dagger, x, a);
    }
}


void
dslashXpayDCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
		int oddBit, int daggerBit, ParitySpinor x, double a) {
    
  dim3 gridDim(res.volume/llink.blockDim, 1, 1);
  dim3 blockDim(llink.blockDim, 1, 1);

  bindFatLongLinkTex(flink, llink, oddBit);
  
  int spinor_bytes = res.length*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);

  if (llink.precision == QUDA_DOUBLE_PRECISION) {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashDD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashDD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashDD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashDD18XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDD18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }

      }
  } else if (llink.precision == QUDA_SINGLE_PRECISION) {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashSD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashSD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashSD18XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSD18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }
  } else {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashHD12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashHD12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashHD8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashHD8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);	  
      }
  }
  
}

void 
dslashXpaySCuda(ParitySpinor res, FullGauge fatlink, FullGauge longlink, ParitySpinor spinor, 
		int oddBit, int daggerBit, ParitySpinor x, double a)
{
    
    dim3 gridDim(res.volume/longlink.blockDim, 1, 1);
    dim3 blockDim(longlink.blockDim, 1, 1);
    
    bindFatLongLinkTex(fatlink, longlink, oddBit);
    
    int spinor_bytes = res.length*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
    cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
    
    int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (longlink.precision == QUDA_DOUBLE_PRECISION) {
      if (longlink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashDS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashDS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (longlink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashDS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashDS18XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDS18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  
      }
  } else if (longlink.precision == QUDA_SINGLE_PRECISION) {
      if (longlink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashSS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (longlink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashSS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashSS18XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }else {
	      dslashSS18DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }
  } else {
      if (longlink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashHS12XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashHS12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (longlink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashHS8XpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashHS8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
	  
      }
  }

}




void
dslashXpayHCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
		int oddBit, int daggerBit, ParitySpinor x, double a) 
{
    
  dim3 gridDim(res.volume/llink.blockDim, 1, 1);
  dim3 blockDim(llink.blockDim, 1, 1);
    
  bindFatLongLinkTex(flink, llink, oddBit);
 
  int spinor_bytes = res.length*sizeof(float)/2;  
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, 2*spinor_bytes/spinorSiteSize); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, 2*spinor_bytes/spinorSiteSize); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (llink.precision == QUDA_DOUBLE_PRECISION) {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	  dslashDH12XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	      ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	  dslashDH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	      ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8XpayKernel <<<gridDim, blockDim, shared_bytes>>> 
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashDH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
    }
  } else if (llink.precision == QUDA_SINGLE_PRECISION) {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
		  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	  } else {
	      dslashSH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
		  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	  }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashSH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
      }
  } else {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashHH12DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	  dslashHH8XpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashHH8DaggerXpayKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    }else{
	printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	exit(1);
	
    }
  }

}


static void
dslashAxpyDCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
		int oddBit, int daggerBit, ParitySpinor x, double a) 
{
    
    dim3 gridDim(res.volume/llink.blockDim, 1, 1);
    dim3 blockDim(llink.blockDim, 1, 1);
    
    bindFatLongLinkTex(flink, llink, oddBit);
    
    int spinor_bytes = res.length*sizeof(double);
    cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
    cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
    
    int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(double);
    
    if (llink.precision == QUDA_DOUBLE_PRECISION) {
	if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	    if (!daggerBit) {
		dslashDD12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashDD12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashDD8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDD8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	}else{
	    if (!daggerBit) {
	      dslashDD18AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDD18DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }

	}
  } else if (llink.precision == QUDA_SINGLE_PRECISION) {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSD12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashSD12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashSD8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSD8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashSD18AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSD18DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }
  } else {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashHD12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashHD12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashHD8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashHD8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((double2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);	  
      }
  }

}


static void 
dslashAxpySCuda(ParitySpinor res, FullGauge fatlink, FullGauge longlink, ParitySpinor spinor, 
		int oddBit, int daggerBit, ParitySpinor x, double a)
{
    
    dim3 gridDim(res.volume/longlink.blockDim, 1, 1);
    dim3 blockDim(longlink.blockDim, 1, 1);
    
    bindFatLongLinkTex(fatlink, longlink, oddBit);
    
    int spinor_bytes = res.length*sizeof(float);
    cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
    cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
    
    int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (longlink.precision == QUDA_DOUBLE_PRECISION) {
      if (longlink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashDS12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
      } else {
	      dslashDS12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (longlink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	  dslashDS8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDS8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashDS18AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashDS18DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  
      }
  } else if (longlink.precision == QUDA_SINGLE_PRECISION) {
      if (longlink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSS12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashSS12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (longlink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashSS8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSS8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  if (!daggerBit) {
	      dslashSS18AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashSS18DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }
  } else {
      if (longlink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashHS12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  } else {
	      dslashHS12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      } else if (longlink.reconstruct == QUDA_RECONSTRUCT_8) {
	  if (!daggerBit) {
	      dslashHS8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
	  else {
	      dslashHS8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> ((float2 *)res.spinor, oddBit, a); CUERR;
	  }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
      }
  }

}





void
dslashAxpyHCuda(ParitySpinor res, FullGauge flink, FullGauge llink, ParitySpinor spinor, 
		int oddBit, int daggerBit, ParitySpinor x, double a) 
{
    
  dim3 gridDim(res.volume/llink.blockDim, 1, 1);
  dim3 blockDim(llink.blockDim, 1, 1);
    
  bindFatLongLinkTex(flink, llink, oddBit);
 
  int spinor_bytes = res.length*sizeof(float)/2;  
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, 2*spinor_bytes/spinorSiteSize); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, 2*spinor_bytes/spinorSiteSize); 
  
  int shared_bytes = blockDim.x*SHARED_FLOATS_PER_THREAD*sizeof(float);

  if (llink.precision == QUDA_DOUBLE_PRECISION) {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	if (!daggerBit) {
	  dslashDH12AxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	      ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	} else {
	  dslashDH12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	      ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8AxpyKernel <<<gridDim, blockDim, shared_bytes>>> 
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashDH8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
    }
  } else if (llink.precision == QUDA_SINGLE_PRECISION) {
      if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
	  if (!daggerBit) {
	      dslashSH12AxpyKernel <<<gridDim, blockDim, shared_bytes>>>
		  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	  } else {
	      dslashSH12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
		  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	  }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8AxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashSH8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
      }else{
	  printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	  exit(1);
      }
  } else {
    if (llink.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12AxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      } else {
	dslashHH12DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
	  ((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
      }
    } else if (llink.reconstruct == QUDA_RECONSTRUCT_8) {
	if (!daggerBit) {
	    dslashHH8AxpyKernel <<<gridDim, blockDim, shared_bytes>>>
		((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	} else {
	    dslashHH8DaggerAxpyKernel <<<gridDim, blockDim, shared_bytes>>>
		((short2*)res.spinor, (float*)res.spinorNorm, oddBit, a);  CUERR;
	}
    }else{
	printf("%s: ERROR: this config is not supported, line %d, file %s \n", __FUNCTION__, __LINE__, __FILE__);
	exit(1);
    }
  }

}




void dslashAxpyCuda(ParitySpinor out, FullGauge fatlink, FullGauge longlink, 
		    ParitySpinor in,  int parity, int dagger,
		    ParitySpinor x, double a) 
{
    if (!initDslash) initDslashCuda(fatlink);
    checkSpinor(in, out);
    checkGaugeSpinor(in, fatlink);
    checkGaugeSpinor(in, longlink);
    
    if (in.precision == QUDA_DOUBLE_PRECISION) {
	dslashAxpyDCuda(out, fatlink, longlink, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_SINGLE_PRECISION) {
	dslashAxpySCuda(out, fatlink, longlink, in, parity, dagger, x, a);
    } else if (in.precision == QUDA_HALF_PRECISION) {
	dslashAxpyHCuda(out, fatlink, longlink, in, parity, dagger, x, a);
    }
}


void 
dslashAxpyFullCuda(FullSpinor out, FullGauge fatlink, FullGauge longlink, 
		   FullSpinor in,  int dagger,
		   FullSpinor x, double a) 
{
    if (!initDslash) initDslashCuda(fatlink);
    checkSpinor(in.even, out.even);
    checkGaugeSpinor(in.even, fatlink);
    checkGaugeSpinor(in.even, longlink);
    
    if (in.even.precision == QUDA_DOUBLE_PRECISION) {
	dslashAxpyDCuda(out.odd, fatlink, longlink, in.even, 1, dagger, x.odd, a);
	dslashAxpyDCuda(out.even, fatlink, longlink, in.odd, 0, dagger, x.even, a);
    } else if (in.even.precision == QUDA_SINGLE_PRECISION) {
	dslashAxpySCuda(out.odd, fatlink, longlink, in.even, 1, dagger, x.odd, a);
	dslashAxpySCuda(out.even, fatlink, longlink, in.odd, 0, dagger, x.even, a);
    } else if (in.even.precision == QUDA_HALF_PRECISION) {
	dslashAxpyHCuda(out.odd, fatlink, longlink, in.even, 1, dagger, x.odd, a);
	dslashAxpyHCuda(out.even, fatlink, longlink, in.odd, 0, dagger, x.even, a);
    }
}

int dslashCudaSharedBytes(Precision precision, int blockDim) {
  if (precision == QUDA_DOUBLE_PRECISION) return blockDim*SHARED_FLOATS_PER_THREAD*sizeof(double);
  else return blockDim*SHARED_FLOATS_PER_THREAD*sizeof(float);
}

// Apply the even-odd preconditioned Dirac operator
void MatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, double kappa, 
	       ParitySpinor tmp, MatPCType matpc_type, int dagger) {

  double kappa2 = -kappa*kappa;
  if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
    dslashCuda(tmp, gauge, in, 1, dagger);
    dslashXpayCuda(out, gauge, tmp, 0, dagger, in, kappa2); 
  } else {
    dslashCuda(tmp, gauge, in, 0, dagger);
    dslashXpayCuda(out, gauge, tmp, 1, dagger, in, kappa2); 
  }
}

void MatPCCuda_st(ParitySpinor out, FullGauge fatlink, FullGauge longlink, ParitySpinor in, double kappa, 
	       ParitySpinor tmp, MatPCType matpc_type, int dagger) 
{
    double kappa2 = -kappa*kappa;
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
	dslashCuda_st(tmp, fatlink, longlink, in, 1, dagger);
	dslashXpayCuda_st(out, fatlink, longlink, tmp, 0, dagger, in, kappa2); 
    } else {
	dslashCuda_st(tmp, fatlink, longlink, in, 0, dagger);
	dslashXpayCuda_st(out, fatlink, longlink, tmp, 1, dagger, in, kappa2); 
    }
}

void MatPCDagMatPCCuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
		       double kappa, ParitySpinor tmp, MatPCType matpc_type) {
  MatPCCuda(out, gauge, in, kappa, tmp, matpc_type, 0);
  MatPCCuda(out, gauge, out, kappa, tmp, matpc_type, 1);
}

void MatPCDagMatPCCuda_st(ParitySpinor out, FullGauge fatlink, FullGauge longlink, ParitySpinor in, 
		       double kappa, ParitySpinor tmp, MatPCType matpc_type) 
{
    struct timeval t0, t1, t2;
    gettimeofday(&t0, NULL); 
    MatPCCuda_st(out, fatlink, longlink, in, kappa, tmp, matpc_type, 0); cudaThreadSynchronize();
    gettimeofday(&t1, NULL);
    MatPCCuda_st(out, fatlink, longlink, out, kappa, tmp, matpc_type, 1); cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    
    //printf("first MatPCCuda=%f ms, second =%f ms,  ",  1000*TDIFF(t1, t0), 1000*TDIFF(t2, t1)); fflush(stdout);
}


// Apply the full operator
void
MatCuda_st(FullSpinor out, FullGauge fatlink, FullGauge longlink, FullSpinor in, double kappa, int dagger) 
{
    dslashXpayCuda_st(out.odd, fatlink, longlink, in.even, 1, dagger, in.odd, -kappa);
    dslashXpayCuda_st(out.even, fatlink, longlink, in.odd, 0, dagger, in.even, -kappa);
}


void MatDagMatCuda(ParitySpinor out, FullGauge fatlink, FullGauge longlink, ParitySpinor in, 
		   double mass, ParitySpinor tmp, int oddBit)
{
    
    dslashCuda_st(tmp, fatlink, longlink, in, 1 - oddBit, 0);
    dslashAxpyCuda(out, fatlink, longlink, tmp, oddBit, 0, in, mass);
}



#include "llfat_quda.cu"

#include "gauge_force_quda.cu"

#include "fermion_force_quda.cu"
