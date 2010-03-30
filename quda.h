#ifndef _QUDA_H
#define _QUDA_H

#include <cuda_runtime.h>

//#define L1 4 // "x" dimension
//#define L2 4 // "y" dimension
//#define L3 4 // "z" dimension
//#define L4 4 // "time" dimension
//#define L1h (L1/2) // half of the full "x" dimension, useful for even/odd lattice indexing

//#define N (L1*L2*L3*L4) // total number of lattice points
//#define Nh (N/2) // total number of even/odd lattice points


#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3
#define TDOWN 4
#define ZDOWN 5
#define YDOWN 6
#define XDOWN 7
#define OPP_DIR(dir)    (7-(dir))
#define GOES_FORWARDS(dir) (dir<=3)
#define GOES_BACKWARDS(dir) (dir>3)


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

  typedef struct {
    int blockDim; // The size of the thread block to use
    size_t bytes;
    Precision precision;
    int length; // total length
    int volume; // geometric volume (single parity)
    int X[4]; // the geometric lengths (single parity)
    int Nc; // number of colors
    ReconstructType reconstruct;
    ParityGauge odd;
    ParityGauge even;
    double anisotropy;
  } FullGauge;
    
    typedef struct {
	int blockDim; // The size of the thread block to use
	size_t bytes;
	Precision precision;
	int length; // total length
	int volume; // geometric volume (single parity)
	int X[4]; // the geometric lengths (single parity)
	int Nc; // number of colors
	ParityGauge odd;
	ParityGauge even;
	double anisotropy;
    } FullStaple;
    
    typedef struct {
	int blockDim; // The size of the thread block to use
	size_t bytes;
	Precision precision;
	int length; // total length
	int volume; // geometric volume (single parity)
	int X[4]; // the geometric lengths (single parity)
	int Nc; // number of colors
	ParityGauge odd;
	ParityGauge even;
	double anisotropy;
    } FullMom;
    

  typedef struct {
    size_t bytes;
    Precision precision;
    int length;
    int volume;
    int Nc;
    int Ns;
    void *clover; // pointer to clover matrix
    void *cloverInverse; // pointer to inverse of clover matrix
  } ParityClover;

  typedef struct {
    Precision precision;
    ParityClover odd;
    ParityClover even;
  } FullClover;

  typedef struct {
    size_t bytes;
    Precision precision;
    int length; // total length
    int volume; // geometric volume (single parity)
    int X[4]; // the geometric lengths (single parity)
    int Nc; // length of color dimension
    int Ns; // length of spin dimension
    void *spinor; // either (double2*), (float4 *) or (short4 *), depending on precision
    float *spinorNorm; // used only when precision is QUDA_HALF_PRECISION
  } ParitySpinor;

  typedef struct {
    ParitySpinor odd;
    ParitySpinor even;
  } FullSpinor;
  
    typedef struct {
	size_t bytes;
	Precision precision;
	int length; // total length
	int volume; // geometric volume (single parity)
	int X[4]; // the geometric lengths (single parity)
	int Nc; // length of color dimension
	int Ns; // length of spin dimension
	void *data; // either (double2*), (float4 *) or (short4 *), depending on precision
	float *dataNorm; // used only when precision is QUDA_HALF_PRECISION
    } ParityHw;
    
    typedef struct {
	ParityHw odd;
	ParityHw even;
    } FullHw;
    
#ifdef __cplusplus
}
#endif

#define TDIFF(t1, t0) (t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec))

#define CUERR  do{ cudaError_t cuda_err;				\
        if ((cuda_err = cudaGetLastError()) != cudaSuccess) {		\
            fprintf(stderr, "ERROR: CUDA error: %s, line %d, function %s, file %s\n", \
		    cudaGetErrorString(cuda_err),  __LINE__, __FUNCTION__, __FILE__); \
            exit(cuda_err);}}while(0) 

#include <invert_quda.h>
#include <blas_quda.h>
#include <dslash_quda.h>

#endif // _QUDA_H
