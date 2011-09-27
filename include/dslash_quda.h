#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <face_quda.h>
#include <clover_field.h>

enum KernelType {
  INTERIOR_KERNEL = 5,
  EXTERIOR_KERNEL_X = 0,
  EXTERIOR_KERNEL_Y = 1,
  EXTERIOR_KERNEL_Z = 2,
  EXTERIOR_KERNEL_T = 3
};

struct LatticeParam {
  int Vh;
  int Vsh;
  int Vs;
  
  int X1;
  int X2;
  int X3;
  int X4;
  int X1h;
  int X2h;
  
  int X2X1;
  int X3X1;
  int X3X2;
  int X3X2X1;
  int X4X2X1;
  int X4X3X1;
  int X4X3X2;
  int X4X2X1h;
  int X4X3X1h;
  int X4X3X2h;

  int X1m1;
  int X2m1;
  int X3m1;
  int X4m1;
  
  int X1m3;
  int X2m3;
  int X3m3;
  int X4m3;
  
  int X2X1mX1;
  int X3X2X1mX2X1;
  int X4X3X2X1mX3X2X1;
  int X4X3X2X1hmX3X2X1h;
  
  // used for improved staggered
  int X1_3;
  int X2_3;
  int X3_3;
  int X4_3;
  int X2X1_3;
  int X3X2X1_3;

  int X2X1m3X1;
  int X3X2X1m3X2X1;
  int X4X3X2X1m3X3X2X1;
  int X4X3X2X1hm3X3X2X1h;

  // used for domain wall  
  int Ls;

  // used for multi-GPU
  int Vh_2d_max;  
  int ghostFace[QUDA_MAX_DIM];
};

struct DslashParam {
  int sp_stride;
  int ga_stride;
  int cl_stride;
  
  int fat_ga_stride;
  int long_ga_stride;
  float fat_ga_max;
  
  int gauge_fixed;
  
  // single precision constants
  float anisotropy_f;
  float coeff_f;
  float t_boundary_f;
  float pi_f;
  
  // double precision constants
  double anisotropy;
  double t_boundary;
  double coeff;
  
  float2 An2;
  float2 TB2;
  float2 No2;
  
  // Are we processor 0 in time?
  bool Pt0;
  
  // Are we processor Nt-1 in time?
  bool PtNm1;
  
  // equal to either 1 or 2; used for temporal spin projection
  double tProjScale;
  float tProjScale_f;
  
  int threads; // the desired number of active threads
  int parity;  // Even-Odd or Odd-Even
  int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
  int ghostDim[QUDA_MAX_DIM]; // Whether a ghost zone has been allocated for a given dimension
  int ghostOffset[QUDA_MAX_DIM];
  int ghostNormOffset[QUDA_MAX_DIM];
  KernelType kernel_type; //is it INTERIOR_KERNEL, EXTERIOR_KERNEL_X/Y/Z/T
};

void setFace(const FaceBuffer &face);

void initCache();
void createStreams();
void destroyStreams();

void setDslashTuning(QudaTune tune);

void setLatticeParam(LatticeParam &latParam, const FullGauge &gauge);
void setDomainWallParam(LatticeParam &latParam, const int Ls);
void setDslashParam(DslashParam &param, const FullGauge &gauge, const int sp_stride);
void setCloverParam(DslashParam &param, const int cl_stride);
void setStaggeredParam(DslashParam &param, const FullGauge &fatgauge, const FullGauge &longgauge);

// plain Wilson Dslash  
void wilsonDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in,
		      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
		      const double &k, const dim3 *block, const int *commDim,
		      const LatticeParam &latParam, DslashParam &dslashParam);

// clover Dslash
void cloverDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, 
		      const FullClover cloverInv, const cudaColorSpinorField *in, 
		      const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
		      const double &k, const dim3 *block, const int *commDim,
		      const LatticeParam &latParam, DslashParam &dslashParam);

// solo clover term
void cloverCuda(cudaColorSpinorField *out, const FullGauge gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int oddBit, const dim3 &block, const int *commDim,
		const DslashParam &dslashParam);

// domain wall Dslash  
void domainWallDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in, 
			  const int parity, const int dagger, const cudaColorSpinorField *x, 
			  const double &m_f, const double &k, const dim3 *blockDim,
			  const LatticeParam &latParam, DslashParam &dslashParam);

// staggered Dslash    
void staggeredDslashCuda(cudaColorSpinorField *out, const FullGauge fatGauge, FullGauge longGauge,
			 const cudaColorSpinorField *in, const int parity, const int dagger, 
			 const cudaColorSpinorField *x, const double &k, const dim3 *block, const int *commDim,
			 const LatticeParam &latParam, DslashParam &dslashParam);

// twisted mass Dslash  
void twistedMassDslashCuda(cudaColorSpinorField *out, const FullGauge gauge, const cudaColorSpinorField *in,
			   const int parity, const int dagger, const cudaColorSpinorField *x, 
			   const double &kappa, const double &mu, const double &a, 
			   const dim3 *block, const int *commDim,
			   const LatticeParam &latParam, DslashParam &dslashParam);

// solo twist term
void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const int dagger, const double &kappa, const double &mu,
		     const QudaTwistGamma5Type, const dim3 &block, DslashParam &dslashParam);

// face packing routines
void packFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dim, const QudaDirection dir, const int dagger, 
		    const int parity, const LatticeParam &latParam, const DslashParam &dslashParam, const cudaStream_t &stream);

#endif // _DSLASH_QUDA_H
