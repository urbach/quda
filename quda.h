//J  quad_dwf.h
//J  Ver. 09.07.b

//J  This replaces quda.h.  Eventually we may want
//J  to merge it with quda.h.  I tried to only modify
//J  s.t. DD_DWF must be defined for the modifications
//J  to take effect.

#ifndef _QUDA_DWF_H
#define _QUDA_DWF_H

// Nvidia file.
#include <cuda_runtime.h>


#define L1 8 // "x" dimension
#define L2 8 // "y" dimension
#define L3 8 // "z" dimension
#define L4 32 // "time" dimension
#define Ls 16 // "s" dimension
#define L1h (L1/2) // half of the full "x" dimension, useful for even/odd lattice indexing

//J  Important:  the gauge fields and spinor fields see a different
//J  number of lattice sites, due to A_\mu indep. of s.  Had to make
//J  a decision how to handle this.  Decided to not define N or Nh,
//J  so that we are forced to look at each instance
//J  where those things were used in Wilson code,
//J  and to contemplate how to modify.
//J  Thus I define N_4d, Nh_4d, N_5d, Nh_5d instead.
#define N_4d (L1*L2*L3*L4) // total number of lattice points
#define Nh_4d (L1h*L2*L3*L4) // total number of even/odd lattice points
#define N_5d (L1*L2*L3*L4*Ls) // total number of lattice points
// Nh_5d is the total number of threads.
#define Nh_5d (L1h*L2*L3*L4*Ls) // total number of even/odd lattice points

#define MAX_SHORT 32767

// The Quda is added to avoid collisions with other libs
#define GaugeFieldOrder QudaGaugeFieldOrder
#define DiracFieldOrder QudaDiracFieldOrder
#define InverterType QudaInverterType  
#define Precision QudaPrecision
#define MatPCType QudaMatPCType
#define SolutionType QudaSolutionType
#define MassNormalization QudaMassNormalization
#define PreserveSource QudaPreserveSource
#define ReconstructType QudaReconstructType
#define GaugeFixed QudaGaugeFixed
#define DagType QudaDagType
#define Tboundary QudaTboundary

#include <enum_quda.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef void *ParityGauge;
  typedef void *ParityClover;

  typedef struct {
    size_t packedGaugeBytes;
    Precision precision;
    ReconstructType reconstruct;
    ParityGauge odd;
    ParityGauge even;
  } FullGauge;
  

  typedef struct {
    Precision precision;
    int length; // geometric length of spinor
    void *spinor; // either (double2*), (float4 *) or (short4 *), depending on precision
    float *spinorNorm; // used only when precision is QUDA_HALF_PRECISION
  } ParitySpinor;

  typedef struct {
    ParitySpinor odd;
    ParitySpinor even;
  } FullSpinor;
  
#ifdef __cplusplus
}
#endif

#include <invert_quda.h>
#include <blas_quda.h>
#include <dslash_quda.h>

#endif // _QUDA_H
