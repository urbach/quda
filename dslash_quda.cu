// dslash_quda.cu
// Ver. 09.12.a

// 10/28/09:  Enabling single precision.
// 11/2/09:  Turning on Mat and MatPC.
// 11/3/09:  Straightened out -kappa in Mat vs. -kappa*kappa in MatPC.
//   Need to check against prototyping.
// 12/1/09:  Checking, understanding, minor fixes.
// 12/5/09:  Checking, understanding, questions documented.  Stopped at HERE.

#include <stdlib.h>
#include <stdio.h>

#include <dslash_quda.h>


// ----------------------------------------------------------------------
// Cuda code

#if (__CUDA_ARCH__ == 130)
static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
    int4 v = tex1Dfetch(t,i);
    return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#endif

// Double precision gauge field
texture<int4, 1> gauge0TexDouble;
texture<int4, 1> gauge1TexDouble;

// Single precision gauge field
texture<float4, 1, cudaReadModeElementType> gauge0TexSingle;
texture<float4, 1, cudaReadModeElementType> gauge1TexSingle;

// Half precision gauge field
texture<short4, 1, cudaReadModeNormalizedFloat> gauge0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> gauge1TexHalf;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<float, 1, cudaReadModeElementType> spinorTexNorm;

// Double precision accumulate spinor field
texture<int4, 1> accumTexDouble;

// Single precision accumulate spinor field
texture<float4, 1, cudaReadModeElementType> accumTexSingle;

// Half precision accumulate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> accumTexHalf;
texture<float, 1, cudaReadModeElementType> accumTexNorm;

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;

QudaGaugeParam *gauge_param;
QudaInvertParam *invert_param;

__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;
__constant__ int X1h;

__constant__ int gauge_fixed;

// single precision constants
__constant__ float anisotropy_f;
__constant__ float t_boundary_f;
__constant__ float pi_f;

// double precision constants
__constant__ double anisotropy;
__constant__ double t_boundary;

#include <dslash_dwf_def.h>

void setCudaGaugeParam() {
  int gf = (gauge_param->gauge_fix == QUDA_GAUGE_FIXED_YES) ? 1 : 0;
  cudaMemcpyToSymbol("gauge_fixed", &(gf), sizeof(int));

  cudaMemcpyToSymbol("anisotropy", &(gauge_param->anisotropy), sizeof(double));

  double t_bc = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary", &(t_bc), sizeof(double));

  float anisotropy_f = gauge_param->anisotropy;
  cudaMemcpyToSymbol("anisotropy_f", &(anisotropy_f), sizeof(float));

  float t_bc_f = (gauge_param->t_boundary == QUDA_PERIODIC_T) ? 1.0 : -1.0;
  cudaMemcpyToSymbol("t_boundary_f", &(t_bc_f), sizeof(float));

  float h_pi_f = M_PI;
  cudaMemcpyToSymbol("pi_f", &(h_pi_f), sizeof(float));
}

//ok
void bindGaugeTex(FullGauge gauge, int oddBit) {
  int reconstruct = (gauge.reconstruct == QUDA_RECONSTRUCT_12) ? 12 : 8;
  int packed_gauge_bytes = 4*Nh_4d*reconstruct;

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    packed_gauge_bytes *= sizeof(double);
    if (oddBit) {
      cudaBindTexture(0, gauge0TexDouble, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexDouble, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexDouble, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexDouble, gauge.odd, packed_gauge_bytes); 
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    packed_gauge_bytes *= sizeof(float);
    if (oddBit) {
      cudaBindTexture(0, gauge0TexSingle, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexSingle, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexSingle, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexSingle, gauge.odd, packed_gauge_bytes); 
    }
  } else {
    packed_gauge_bytes *= sizeof(float)/2;
    if (oddBit) {
      cudaBindTexture(0, gauge0TexHalf, gauge.odd, packed_gauge_bytes); 
      cudaBindTexture(0, gauge1TexHalf, gauge.even, packed_gauge_bytes);
    } else {
      cudaBindTexture(0, gauge0TexHalf, gauge.even, packed_gauge_bytes);
      cudaBindTexture(0, gauge1TexHalf, gauge.odd, packed_gauge_bytes); 
    }
  }
}


// ----------------------------------------------------------------------

void checkSpinor(ParitySpinor out, ParitySpinor in) {
  if (in.precision != out.precision) {
    printf("Error in dslash quda: input and out spinor precisions don't match\n");
    exit(-1);
  }
}

//J  Small name changes to allow for dwf functions.
//ok
void dslash_dwf_Cuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
    int parity, int dagger, double mferm) {
    checkSpinor(in, out);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
#ifndef NO_D_PREC
    dslashD_dwf_Cuda(out, gauge, in, parity, dagger,mferm);
#endif    
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
#ifndef NO_S_PREC
    dslashS_dwf_Cuda(out, gauge, in, parity, dagger,mferm);
#endif
  } else if (in.precision == QUDA_HALF_PRECISION) {
#ifndef NO_H_PREC
    dslashH_dwf_Cuda(out, gauge, in, parity, dagger,mferm);
#endif
  }
}

#ifndef NO_D_PREC      
//ok
void dslashD_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit, double mferm) {
  
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);


  int spinor_dwf_bytes = Nh_5d*spinorSiteSize*sizeof(double);

  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_dwf_bytes); 

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
         // These Kernels are defined as inline functions in dslash_dwf_def.h.
         // That may not be apparent b/c it is iterative at this point.
      	dslashDD12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,
            mferm);
      } else {
	      dslashDD12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,
            mferm);
      }
    } else {

      if (!daggerBit) {
      	dslashDD8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,
            mferm);
      } else {
	      dslashDD8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,
            mferm);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {

    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_S_PREC
	dslashSD12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
      } else {
	dslashSD12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
#endif
      }
    } else {
      if (!daggerBit) {
#ifndef NO_S_PREC
	dslashSD8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
      } else {
	dslashSD8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
#endif
      }
    }
  } else {

    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_H_PREC
	dslashHD12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
      } else {
	dslashHD12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
#endif
      }
    } else {
      if (!daggerBit) {
#ifndef NO_H_PREC
	dslashHD8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
#endif
      } else {
#ifndef NO_H_PREC
	dslashHD8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((double2 *)res.spinor, oddBit,mferm);
#endif
      }
    }
  }
}
#endif

#ifndef NO_S_PREC
//ok
void dslashS_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit,double mferm) {
  
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float);
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 

  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_D_PREC      
	dslashDS12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif
      } else {
#ifndef NO_D_PREC      
	dslashDS12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif
      }
    } else {
      if (!daggerBit) {
#ifndef NO_D_PREC      
	dslashDS8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif
      } else {
#ifndef NO_D_PREC      
	dslashDS8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSS12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
      } else {
	dslashSS12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
      }
    } else {
      if (!daggerBit) {
	dslashSS8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
      } else {
	dslashSS8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_H_PREC      
	dslashHS12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif
      } else {
#ifndef NO_H_PREC      
	dslashHS12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif
      }
    } else {
      if (!daggerBit) {
#ifndef NO_H_PREC      
	dslashHS8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif  
      } else {
#ifndef NO_H_PREC      
	dslashHS8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm);
#endif  
      }
    }
  }
}
#endif

#ifndef NO_H_PREC
void dslashH_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		 int oddBit, int daggerBit) {

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashDH12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashDH8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashDH8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashSH12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashSH8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashSH8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashHH12Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    } else {
      if (!daggerBit) {
	dslashHH8_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      } else {
	dslashHH8Dagger_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((short4*)res.spinor, (float*)res.spinorNorm, oddBit);
      }
    }
  }
  
}
#endif

//ok
void dslashXpay_dwf_Cuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, int parity, int dagger,
		    ParitySpinor x, double mferm, double a) {
  checkSpinor(in, out);

  if (in.precision == QUDA_DOUBLE_PRECISION) {
#ifndef NO_D_PREC
    dslashXpayD_dwf_Cuda(out, gauge, in, parity, dagger, x, mferm, a);
#endif
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
#ifndef NO_S_PREC  
    dslashXpayS_dwf_Cuda(out, gauge, in, parity, dagger, x, mferm, a);
#endif
  } else if (in.precision == QUDA_HALF_PRECISION) {
#ifndef NO_H_PREC
    dslashXpayH_dwf_Cuda(out, gauge, in, parity, dagger, x, mferm, a);
#endif
  }
}

#ifndef NO_D_PREC
//ok
void dslashXpayD_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		     int oddBit, int daggerBit, ParitySpinor x, double mferm, double a) {
  // In the preconditioned case, "a" comes in equal to -kappa^2.
  // I'm a bit confused why they use the same name as the
  // builtin variable here.  Doesn't that just lead to a
  // local vs. global variable name conflict?
  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);

  bindGaugeTex(gauge, oddBit);

  // Bind half spinors to textures to allow cached access.  Doesn't
  // this only help if 2 threads access the same data?  But for the
  // diagonal operations of xpay, I don't presently see why that
  // would ever happen.  TODO : test a version w/o these texture
  // binding and see what happens to performance.
  int spinor_bytes = Nh_5d*spinorSiteSize*sizeof(double);
  cudaBindTexture(0, spinorTexDouble, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexDouble, x.spinor, spinor_bytes); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_D_PREC   
  // Declared in dslash_dwf_def.h
  // 11/2/09:  Checked that this code matches the declaration.   
	dslashDD12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, 
    oddBit, mferm, a);
#endif
      } else {
#ifndef NO_D_PREC      
	dslashDD12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, 
    oddBit, mferm, a);
#endif
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
#ifndef NO_D_PREC      
	dslashDD8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, 
    mferm, a);
#endif
      }
      else {
#ifndef NO_D_PREC      
	dslashDD8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit,
    mferm, a);
#endif
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_S_PREC      
	dslashSD12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit,mferm, a);
      } else {
	dslashSD12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit,mferm, a);
#endif
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
#ifndef NO_S_PREC
	dslashSD8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit,mferm, a);
      } else {
	dslashSD8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit,mferm, a);
#endif
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_H_PREC      
	dslashHD12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, mferm, a);
      } else {
	dslashHD12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, mferm, a);
#endif
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
#ifndef NO_H_PREC      
	dslashHD8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, mferm, a);
      } else {
	dslashHD8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_DOUBLE>>> ((double2 *)res.spinor, oddBit, mferm, a);
#endif
      }
    }
  }
}
#endif

#ifndef NO_S_PREC
//ok
void dslashXpayS_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, double mferm, double a) {

  dim3 gridDim(GRID_DIM, 1, 1);  // GRID_DIM defined in dslash_quda.h
  dim3 blockDim(BLOCK_DIM, 1, 1);  // BLOCK_DIM also defined there, currently
     // set equal to 64.  That is, 64 threads per block.

  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float);  // Note that this is
    // a half spinor size.
  cudaBindTexture(0, spinorTexSingle, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexSingle, x.spinor, spinor_bytes); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_D_PREC      
	dslashDS12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
#endif
      } else {
#ifndef NO_D_PREC      
	dslashDS12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
#endif
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
#ifndef NO_D_PREC      
	dslashDS8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
#endif
      }
      else {
#ifndef NO_D_PREC      
	dslashDS8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
#endif
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {

        // This is where it is currently going.  Make sure this is right.
        // Pass DWF height in param a.  Look at dslash_def.h file.  Seem to
        // match prototype w/ macros DD_PARAM1 and DD_PARAM2.
	dslashSS12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
      } else {
        // This is where it is currently going.  Make sure this is right.
        // see just above
	dslashSS12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSS8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm, a);
      }
      else {
	dslashSS8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
#ifndef NO_H_PREC      
	dslashHS12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
#endif
    } else {
#ifndef NO_H_PREC      
	dslashHS12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm, a);
#endif
    }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
#ifndef NO_H_PREC      
	dslashHS8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit, mferm,a);
#endif
      }
      else {
#ifndef NO_H_PREC      
	dslashHS8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> ((float4 *)res.spinor, oddBit,mferm, a);
#endif
      }
    }
  }
}
#endif

#ifndef NO_H_PREC
void dslashXpayH_dwf_Cuda(ParitySpinor res, FullGauge gauge, ParitySpinor spinor, 
		    int oddBit, int daggerBit, ParitySpinor x, double mferm, double a) {

  dim3 gridDim(GRID_DIM, 1, 1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
    
  bindGaugeTex(gauge, oddBit);

  int spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float)/2;
  cudaBindTexture(0, spinorTexHalf, spinor.spinor, spinor_bytes); 
  cudaBindTexture(0, spinorTexNorm, spinor.spinorNorm, spinor_bytes/12); 
  cudaBindTexture(0, accumTexHalf, x.spinor, spinor_bytes); 
  cudaBindTexture(0, accumTexNorm, x.spinorNorm, spinor_bytes/12); 
  
  if (gauge.precision == QUDA_DOUBLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashDH12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashDH12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashDH8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>> 
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashDH8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  } else if (gauge.precision == QUDA_SINGLE_PRECISION) {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashSH12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashSH12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashSH8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashSH8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  } else {
    if (gauge.reconstruct == QUDA_RECONSTRUCT_12) {
      if (!daggerBit) {
	dslashHH12Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashHH12DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    } else if (gauge.reconstruct == QUDA_RECONSTRUCT_8) {
      if (!daggerBit) {
	dslashHH8Xpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      } else {
	dslashHH8DaggerXpay_dwf_Kernel <<<gridDim, blockDim, SHARED_BYTES_SINGLE>>>
	  ((short4*)res.spinor, (float*)res.spinorNorm, oddBit, a);
      }
    }
  }

}
#endif

int dslashCudaSharedBytes() {
  return SHARED_BYTES_SINGLE;
}

// Apply the even-odd preconditioned Dirac operator
//  mferm = fermion mass.
//  m0_dwf = dwf barrier height -- hidden in kappa.
// This host function calls other host functions, depending on
// level of precision, which then call the corresponding device kernels.
// The offdiag dslash is combined with the diagonal+offdiag XpayD to get
// the total MatPC.
//ok
void MatPC_dwf_Cuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, double kappa, 
  ParitySpinor tmp, MatPCType matpc_type, int dagger, double mferm) {

  // Make sure precisions match.
  checkSpinor(in, out);
  checkSpinor(in, tmp);

  // The diagonal term coefficient, for 5d PC.
  double kappa2= -kappa*kappa;

  if (in.precision == QUDA_DOUBLE_PRECISION) {  
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      // Hopping terms.  oddBit=1.
      // This is a modification to what was dslashDCuda() in the
      // Wilson code.  It takes an extra argument, since for DWF
      // the fermion mass goes into the hopping terms.
#ifndef NO_D_PREC      
      dslashD_dwf_Cuda(tmp, gauge, in, 1, dagger, mferm);  // Line 171 above.
      // Diagonal terms.  Look at dslashXpayD_dwf_Cuda for model.
      // This doesn't seem to require a modification to what 
      // was dslashXpayDCuda() in the Wilson code.  I am passing
      // mferm in the diagonal terms of DWF b/c xpayD does a
      // dslash AND the diagonal term.  More precisely, for oddBit=0 it does:
      //   out_e = -kappa^2 D_{eo} tmp_o + in_e
      // through commands like:
      //   o00_re = a*o00_re + accum0.x;
      // Here accum0.x comes from "in" (via texture binding), while "tmp" is the input spinor
      // that gets operated on by the 5d dslash code, producing o00_re.
      // Since "tmp" is the output of the dslash above, what we really
      // end up with is:
      //   out_e = ( 1 - kappa^2 D_{eo} D_{oe} ) in_e
      // which is the usual 5dPC.  
      dslashXpayD_dwf_Cuda(out, gauge, tmp, 0, dagger, in, mferm, kappa2); // Line 405 above.
#endif      
    } else {
      // Opposite parity operations.
#ifndef NO_D_PREC      
      dslashD_dwf_Cuda(tmp, gauge, in, 0, dagger, mferm);
      dslashXpayD_dwf_Cuda(out, gauge, tmp, 1, dagger, in, mferm, kappa2); 
#endif      
    }
  } else if (in.precision == QUDA_SINGLE_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
#ifndef NO_S_PREC    
      dslashS_dwf_Cuda(tmp, gauge, in, 1, dagger, mferm);
      dslashXpayS_dwf_Cuda(out, gauge, tmp, 0, dagger, in, mferm, kappa2); //Line 487 above.
#endif      
    } else {
#ifndef NO_S_PREC    
      dslashS_dwf_Cuda(tmp, gauge, in, 0, dagger, mferm);
      dslashXpayS_dwf_Cuda(out, gauge, tmp, 1, dagger, in, mferm, kappa2); 
#endif
    }
  } else if (in.precision == QUDA_HALF_PRECISION) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
#ifndef NO_H_PREC    
      dslashH_dwf_Cuda(tmp, gauge, in, 1, dagger, mferm);
      dslashXpayH_dwf_Cuda(out, gauge, tmp, 0, dagger, in, mferm, kappa2); 
#endif      
    } else {
#ifndef NO_H_PREC    
      dslashH_dwf_Cuda(tmp, gauge, in, 0, dagger, mferm);
      dslashXpayH_dwf_Cuda(out, gauge, tmp, 1, dagger, in, mferm, kappa2); 
#endif      
    }
  }

}

// matpc_type says whether even-even or odd-odd
//ok
void MatPCDagMatPC_dwf_Cuda(ParitySpinor out, FullGauge gauge, ParitySpinor in, 
		       double kappa, ParitySpinor tmp, MatPCType matpc_type, double mferm) {
  MatPC_dwf_Cuda(out, gauge, in, kappa, tmp, matpc_type, 0, mferm);
  MatPC_dwf_Cuda(out, gauge, out, kappa, tmp, matpc_type, 1, mferm);
}


// Apply the full operator
//ok
void Mat_dwf_Cuda(FullSpinor out, FullGauge gauge, FullSpinor in, 
   double kappa, int dagger, double mferm) 
{
  // Check that precisions match.
  checkSpinor(in.even, out.even);

  if (in.even.precision == QUDA_DOUBLE_PRECISION) {
#ifndef NO_D_PREC    
    // out.odd = -kappa D_{oe} in.even + in.odd
    // Defined in this file, near Line 407.
    dslashXpayD_dwf_Cuda(out.odd, gauge, in.even, 1, dagger, in.odd, mferm, -kappa);
    // out.even = -kappa D_{eo} in.odd + in.even
    dslashXpayD_dwf_Cuda(out.even, gauge, in.odd, 0, dagger, in.even, mferm, -kappa);
#endif
  } else if (in.even.precision == QUDA_SINGLE_PRECISION) {
#ifndef NO_S_PREC    
    dslashXpayS_dwf_Cuda(out.odd, gauge, in.even, 1, dagger, in.odd, mferm, -kappa);
    dslashXpayS_dwf_Cuda(out.even, gauge, in.odd, 0, dagger, in.even, mferm, -kappa);
#endif
  } else if (in.even.precision == QUDA_HALF_PRECISION) {
#ifndef NO_H_PREC    
    dslashXpayH_dwf_Cuda(out.odd, gauge, in.even, 1, dagger, in.odd, mferm, -kappa);
    dslashXpayH_dwf_Cuda(out.even, gauge, in.odd, 0, dagger, in.even, mferm, -kappa);
#endif    
  }
}



