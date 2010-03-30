#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <util_quda.h>

#define REDUCE_THREADS 128
#define REDUCE_MAX_BLOCKS 64

#define REDUCE_DOUBLE 64
#define REDUCE_KAHAN 32

#define REDUCE_TYPE REDUCE_DOUBLE
#define QudaSumFloat double
#define QudaSumComplex cuDoubleComplex
#define QudaSumFloat3 double3

// These are used for reduction kernels
QudaSumFloat *d_reduceFloat;
QudaSumComplex *d_reduceComplex;
QudaSumFloat3 *d_reduceFloat3;

QudaSumFloat h_reduceFloat[REDUCE_MAX_BLOCKS];
QudaSumComplex h_reduceComplex[REDUCE_MAX_BLOCKS];
QudaSumFloat3 h_reduceFloat3[REDUCE_MAX_BLOCKS];

int blocksFloat = 0;
int blocksComplex = 0;
int blocksFloat3 = 0;

void initReduceFloat(int blocks) {
  if (blocks != blocksFloat) {
    if (blocksFloat > 0) cudaFree(d_reduceFloat);

    if (cudaMalloc((void**) &d_reduceFloat, blocks*sizeof(QudaSumFloat))) {
      printf("Error allocating reduction matrix\n");
      exit(0);
    }

    blocksFloat = blocks;

  }
}

void initReduceComplex(int blocks) {
  if (blocks != blocksComplex) {
    if (blocksComplex > 0) cudaFree(d_reduceComplex);

    if (cudaMalloc((void**) &d_reduceComplex, blocks*sizeof(QudaSumComplex))) {
      printf("Error allocating reduction matrix\n");
      exit(0);
    }

    blocksComplex = blocks;
    printf("Initialized reduce complex %d\n", blocksComplex);
  }
}

void initReduceFloat3(int blocks) {
  if (blocks != blocksFloat3) {
    if (blocksFloat3 > 0) cudaFree(d_reduceFloat3);

    if (cudaMalloc((void**) &d_reduceFloat3, blocks*sizeof(QudaSumFloat3))) {
      printf("Error allocating reduction matrix\n");
      exit(0);
    }

    blocksFloat3 = blocks;
    printf("Initialized reduce float3 %d\n", blocksFloat3);
  }
}

static __inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}



#define RECONSTRUCT_HALF_SPINOR(a, texHalf, texNorm, length)		\
    float2 a##0 = tex1Dfetch(texHalf, i + 0*length);			\
    float2 a##1 = tex1Dfetch(texHalf, i + 1*length);			\
    float2 a##2 = tex1Dfetch(texHalf, i + 2*length);			\
    {float b = tex1Dfetch(texNorm, i);					\
    (a##0).x *= b; (a##0).y *= b;					\
    (a##1).x *= b; (a##1).y *= b;					\
    (a##2).x *= b; (a##2).y *= b;}

#define CONSTRUCT_HALF_SPINOR_FROM_SINGLE(h, n, a, length)		\
    {float c0 = fmaxf(fabsf((a##0).x), fabsf((a##0).y));		\
	float c1 = fmaxf(fabsf((a##1).x), fabsf((a##1).y));		\
	float c2 = fmaxf(fabsf((a##2).x), fabsf((a##2).y));		\
	c0 = fmaxf(c0, c1); c0 = fmaxf(c0, c2);				\
	n[i] = c0;							\
	float C = __fdividef(MAX_SHORT, c0);				\
	h[i+0*length] = make_short2((short)(C*(float)(a##0).x), (short)(C*(float)(a##0).y)); \
	h[i+1*length] = make_short2((short)(C*(float)(a##1).x), (short)(C*(float)(a##1).y)); \
	h[i+2*length] = make_short2((short)(C*(float)(a##2).x), (short)(C*(float)(a##2).y));}

#define CONSTRUCT_HALF_SPINOR_FROM_DOUBLE(h, n, a, length)		\
    {float c0 = fmaxf(fabsf((a##0).x), fabsf((a##0).y));		\
	float c1 = fmaxf(fabsf((a##1).x), fabsf((a##1).y));		\
	float c2 = fmaxf(fabsf((a##2).x), fabsf((a##2).y));		\
	c0 = fmaxf(c0, c1); c0 = fmaxf(c0, c2);				\
	n[i] = c0;							\
	float C = __fdividef(MAX_SHORT, c0);				\
	h[i+0*length] = make_short2((short)(C*(float)(a##0).x), (short)(C*(float)(a##0).y)); \
	h[i+1*length] = make_short2((short)(C*(float)(a##1).x), (short)(C*(float)(a##1).y)); \
	h[i+2*length] = make_short2((short)(C*(float)(a##2).x), (short)(C*(float)(a##2).y));}

#define SUM_FLOAT4(sum, a)			\
  float sum = a.x + a.y + a.z + a.w;

#define REAL_DOT_FLOAT4(dot, a, b) \
  float dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w

#define REAL_DOT_FLOAT2(dot, a, b)		\
    float dot = a.x*b.x + a.y*b.y

#define IMAG_DOT_FLOAT4(dot, a, b) \
  float dot = a.x*b.y - a.y*b.x + a.z*b.w - a.w*b.z

#define AX_FLOAT4(a, X)				\
  X.x *= a; X.y *= a; X.z *= a; X.w *= a;

#define AX_FLOAT2(a, X)				\
    X.x *= a; X.y *= a;

#define XPY_FLOAT4(X, Y)		     \
  Y.x += X.x; Y.y += X.y; Y.z += X.z; Y.w += X.w;

#define XPY_FLOAT2(X, Y)		     \
    Y.x += X.x; Y.y += X.y; 

#define XMY_FLOAT4(X, Y)		     \
  Y.x = X.x - Y.x; Y.y = X.y - X.y; Y.z = X.z - Y.z; Y.w = X.w - Y.w;

#define XMY_FLOAT2(X, Y)			\
    Y.x = X.x - Y.x; Y.y = X.y - X.y;

#define MXPY_FLOAT4(X, Y)		     \
  Y.x -= X.x; Y.y -= X.y; Y.z -= X.z; Y.w -= X.w;

#define MXPY_FLOAT2(X, Y)		     \
    Y.x -= X.x; Y.y -= X.y; 

#define AXPY_FLOAT4(a, X, Y)		     \
  Y.x += a*X.x;	Y.y += a*X.y;		     \
  Y.z += a*X.z;	Y.w += a*X.w;

#define AXPY_FLOAT2(a, X, Y)			\
    Y.x += a*X.x;	Y.y += a*X.y;		     

#define AXPBY_FLOAT4(a, X, b, Y)		\
  Y.x = a*X.x + b*Y.x; Y.y = a*X.y + b*Y.y;	\
  Y.z = a*X.z + b*Y.z; Y.w = a*X.w + b*Y.w;

#define XPAY_FLOAT4(X, a, Y)			     \
  Y.x = X.x + a*Y.x; Y.y = X.y + a*Y.y;		     \
  Y.z = X.z + a*Y.z; Y.w = X.w + a*Y.w;

#define XPAY_FLOAT2(X, a, Y)			\
    Y.x = X.x + a*Y.x; Y.y = X.y + a*Y.y;		     

#define AXPBY_FLOAT2(a, X, b, Y)			\
    Y.x = a*X.x + b*Y.x; Y.y = a*X.y + b*Y.y;

#define CAXPY_FLOAT4(a, X, Y) \
  Y.x += a.x*X.x - a.y*X.y;   \
  Y.y += a.y*X.x + a.x*X.y;   \
  Y.z += a.x*X.z - a.y*X.w;   \
  Y.w += a.y*X.z + a.x*X.w;

#define CMAXPY_FLOAT4(a, X, Y)			\
  Y.x -= (a.x*X.x - a.y*X.y);			\
  Y.y -= (a.y*X.x + a.x*X.y);			\
  Y.z -= (a.x*X.z - a.y*X.w);			\
  Y.w -= (a.y*X.z + a.x*X.w);

#define CAXPBY_FLOAT4(a, X, b, Y)		 \
  Y.x = a.x*X.x - a.y*X.y + b.x*Y.x - b.y*Y.y;   \
  Y.y = a.y*X.x + a.x*X.y + b.y*Y.x + b.x*Y.y;   \
  Y.z = a.x*X.z - a.y*X.w + b.x*Y.z - b.y*Y.w;   \
  Y.w = a.y*X.z + a.x*X.w + b.y*Y.z + b.x*Y.w;

#define CXPAYPBZ_FLOAT4(X, a, Y, b, Z)		\
  {float2 z;					       \
  z.x = X.x + a.x*Y.x - a.y*Y.y + b.x*Z.x - b.y*Z.y;   \
  z.y = X.y + a.y*Y.x + a.x*Y.y + b.y*Z.x + b.x*Z.y;   \
  Z.x = z.x; Z.y = z.y;				       \
  z.x = X.z + a.x*Y.z - a.y*Y.w + b.x*Z.z - b.y*Z.w;   \
  z.y = X.w + a.y*Y.z + a.x*Y.w + b.y*Z.z + b.x*Z.w;   \
  Z.z = z.x; Z.w = z.y;}

#define CAXPBYPZ_FLOAT4(a, X, b, Y, Z)		  \
  Z.x += a.x*X.x - a.y*X.y + b.x*Y.x - b.y*Y.y;   \
  Z.y += a.y*X.x + a.x*X.y + b.y*Y.x + b.x*Y.y;   \
  Z.z += a.x*X.z - a.y*X.w + b.x*Y.z - b.y*Y.w;   \
  Z.w += a.y*X.z + a.x*X.w + b.y*Y.z + b.x*Y.w;

// Double precision input spinor field
texture<int4, 1> spinorTexDouble;

// Single precision input spinor field
texture<float2, 1, cudaReadModeElementType> spinorTexSingle;

// Half precision input spinor field
texture<short2, 1, cudaReadModeNormalizedFloat> texHalf1;
texture<float, 1, cudaReadModeElementType> texNorm1;

// Half precision input spinor field
texture<short2, 1, cudaReadModeNormalizedFloat> texHalf2;
texture<float, 1, cudaReadModeElementType> texNorm2;

// Half precision input spinor field
texture<short2, 1, cudaReadModeNormalizedFloat> texHalf3;
texture<float, 1, cudaReadModeElementType> texNorm3;

inline void checkSpinor(ParitySpinor &a, ParitySpinor &b) {
  if (a.precision != b.precision) {
    printf("checkSpinor error, precisions do not match: %d %d\n", a.precision, b.precision);
    exit(-1);
  }

  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

// For kernels with precision conversion built in
inline void checkSpinorLength(ParitySpinor &a, ParitySpinor &b) {
  if (a.length != b.length) {
    printf("checkSpinor error, lengths do not match: %d %d\n", a.length, b.length);
    exit(-1);
  }
}

// cuda's floating point format, IEEE-754, represents the floating point
// zero as 4 zero bytes
void zeroCuda(ParitySpinor a) {
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    cudaMemset(a.spinor, 0, a.length*sizeof(double));
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    cudaMemset(a.spinor, 0, a.length*sizeof(float));
  } else {
    cudaMemset(a.spinor, 0, a.length*sizeof(short));
    cudaMemset(a.spinorNorm, 0, a.length*sizeof(float)/spinorSiteSize);
  }
}

void zeroFullCuda(FullSpinor a) 
{
    zeroCuda(a.even);
    zeroCuda(a.odd);

}
__global__ void convertDSKernel(double2 *dst, float2 *src, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
      for (int k=0; k<3; k++) {
	  dst[k*length+i].x = src[k*length+i].x;
	  dst[k*length+i].y = src[k*length+i].y;
      }
      i += gridSize;
  }   
}

__global__ void convertSDKernel(float2 *dst, double2 *src, int length) {
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    for (int k=0; k<3; k++) {
	dst[k*length+i].x = src[k*length+i].x;
	dst[k*length+i].y = src[k*length+i].y;
    }
    i += gridSize;
  }   
}

__global__ void convertHSKernel(short2 *h, float *norm, int length) 
{
    
    int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    
    while(i < length) {
	float2 F0 = tex1Dfetch(spinorTexSingle, i + 0*length);
	float2 F1 = tex1Dfetch(spinorTexSingle, i + 1*length);
	float2 F2 = tex1Dfetch(spinorTexSingle, i + 2*length);
	CONSTRUCT_HALF_SPINOR_FROM_SINGLE(h, norm, F, length);
	i += gridSize;
    }
    
}

__global__ void convertSHKernel(float2 *res, int length) {

    int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    
    while (i<length) {
	RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, length);
	res[0*length+i] = I0;
	res[1*length+i] = I1;
	res[2*length+i] = I2;
	i += gridSize;
    }
}

__global__ void convertHDKernel(short2 *h, float *norm, int length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x; 
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < length) {
      double2 F0 = fetch_double2(spinorTexDouble, i+0*length);
      double2 F1 = fetch_double2(spinorTexDouble, i+1*length);
      double2 F2 = fetch_double2(spinorTexDouble, i+2*length);
      CONSTRUCT_HALF_SPINOR_FROM_DOUBLE(h, norm, F, length);
      i += gridSize;
  }
}

__global__ void convertDHKernel(double2 *res, int length) {

  int i = blockIdx.x*(blockDim.x) + threadIdx.x; 
  unsigned int gridSize = gridDim.x*blockDim.x;

  while(i < length) {
      RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, length);
      res[0*length+i] = make_double2(I0.x, I0.y);
      res[1*length+i] = make_double2(I1.x, I1.y);
      res[2*length+i] = make_double2(I2.x, I2.y);
      i += gridSize;
  }

}

void 
copyCuda(ParitySpinor dst, ParitySpinor src) 
{
    checkSpinorLength(dst, src);

    int convertLength = dst.length / spinorSiteSize;
    int blocks = min(REDUCE_MAX_BLOCKS, max(convertLength/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    if (dst.precision == QUDA_DOUBLE_PRECISION && src.precision == QUDA_SINGLE_PRECISION) {
	convertDSKernel<<<dimGrid, dimBlock>>>((double2*)dst.spinor, (float2*)src.spinor, convertLength);
    } else if (dst.precision == QUDA_SINGLE_PRECISION && src.precision == QUDA_DOUBLE_PRECISION) {
	convertSDKernel<<<dimGrid, dimBlock>>>((float2*)dst.spinor, (double2*)src.spinor, convertLength);
    } else if (dst.precision == QUDA_SINGLE_PRECISION && src.precision == QUDA_HALF_PRECISION) {
	int spinor_bytes = dst.length*sizeof(short);
	cudaBindTexture(0, texHalf1, src.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, src.spinorNorm, 2*spinor_bytes/spinorSiteSize);
	convertSHKernel<<<dimGrid, dimBlock>>>((float2*)dst.spinor, convertLength);
    } else if (dst.precision == QUDA_HALF_PRECISION && src.precision == QUDA_SINGLE_PRECISION) {
	int spinor_bytes = dst.length*sizeof(float);
	cudaBindTexture(0, spinorTexSingle, src.spinor, spinor_bytes); 
	convertHSKernel<<<dimGrid, dimBlock>>>((short2*)dst.spinor, (float*)dst.spinorNorm, convertLength);
    } else if (dst.precision == QUDA_DOUBLE_PRECISION && src.precision == QUDA_HALF_PRECISION) {
	int spinor_bytes = dst.length*sizeof(short);
	cudaBindTexture(0, texHalf1, src.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, src.spinorNorm, spinor_bytes/3);
	convertDHKernel<<<dimGrid, dimBlock>>>((double2*)dst.spinor, convertLength);
    } else if (dst.precision == QUDA_HALF_PRECISION && src.precision == QUDA_DOUBLE_PRECISION) {
	int spinor_bytes = dst.length*sizeof(double);
	cudaBindTexture(0, spinorTexDouble, src.spinor, spinor_bytes); 
	convertHDKernel<<<dimGrid, dimBlock>>>((short2*)dst.spinor, (float*)dst.spinorNorm, convertLength);
    } else if (dst.precision == QUDA_DOUBLE_PRECISION) {
	cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(double), cudaMemcpyDeviceToDevice);
    } else if (dst.precision == QUDA_SINGLE_PRECISION) {
	cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
	cudaMemcpy(dst.spinor, src.spinor, dst.length*sizeof(short), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dst.spinorNorm, src.spinorNorm, dst.length*sizeof(float)/spinorSiteSize, cudaMemcpyDeviceToDevice);
    }
}


void 
copyFullCuda(FullSpinor dst, FullSpinor src) 
{
    copyCuda(dst.even, src.even);
    copyCuda(dst.odd, src.odd);    
}



template <typename Float>
__global__ void axpbyKernel(Float a, Float *x, Float b, Float *y, int length) {
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < length) {
	y[i] = a*x[i] + b*y[i];
	i += gridSize;
    }
}

__global__ void axpbyHKernel(float a, float b, short2 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    AXPBY_FLOAT2(a, x0, b, y0);
    AXPBY_FLOAT2(a, x1, b, y1);
    AXPBY_FLOAT2(a, x2, b, y2);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  }

}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpbyCuda(double a, ParitySpinor x, double b, ParitySpinor y) 
{
    checkSpinor(x, y);
    int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if (x.precision == QUDA_DOUBLE_PRECISION) {
	axpbyKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, b, (double*)y.spinor, x.length);
    } else if (x.precision == QUDA_SINGLE_PRECISION) {
	axpbyKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float)b, (float*)y.spinor, x.length);
    } else {
	int spinor_bytes = x.length*sizeof(short);
	cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes);
	cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);
	cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes);
	cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);
	axpbyHKernel<<<dimGrid, dimBlock>>>((float)a, (float)b, (short2*)y.spinor,
					    (float*)y.spinorNorm, y.length/spinorSiteSize);
    }
}



template <typename Float>
__global__ void xpyKernel(Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] += x[i];
    i += gridSize;
  } 
}

__global__ void xpyHKernel(short2 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    XPY_FLOAT2(x0, y0);
    XPY_FLOAT2(x1, y1);
    XPY_FLOAT2(x2, y2);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = x[i] + y[i]
void xpyCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    xpyKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    xpyKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
    cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
    xpyHKernel<<<dimGrid, dimBlock>>>((short2*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}
void 
xpyFullCuda(FullSpinor x, FullSpinor y)
{
    xpyCuda(x.even, y.even);
    xpyCuda(x.odd, y.odd);

}

template <typename Float>
__global__ void axpyKernel(Float a, Float *x, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] += a*x[i];
    i += gridSize;
  } 
}

__global__ void axpyHKernel(float a, short2 *yH, float *yN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    AXPY_FLOAT2(a, x0, y0);
    AXPY_FLOAT2(a, x1, y1);
    AXPY_FLOAT2(a, x2, y2);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    i += gridSize;
  } 
  
}

// performs the operation y[i] = a*x[i] + y[i]
void axpyCuda(double a, ParitySpinor x, ParitySpinor y) 
{
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
      axpyKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
      axpyKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
  } else {
      int spinor_bytes = x.length*sizeof(short);
      cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
      cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
      cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
      cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
      axpyHKernel<<<dimGrid, dimBlock>>>((float)a, (short2*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

void axpyFullCuda(double a, FullSpinor x, FullSpinor y) 
{
    axpyCuda(a, x.even, y.even);
    axpyCuda(a, x.odd, y.odd);
        
}

template <typename Float>
__global__ void xpayKernel(Float *x, Float a, Float *y, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    y[i] = x[i] + a*y[i];
    i += gridSize;
  } 
}

__global__ void xpayHKernel(float a, short2 *yH, float *yN, int length) 
{
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < length) {
	RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
	RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
	XPAY_FLOAT2(x0, a, y0);
	XPAY_FLOAT2(x1, a, y1);
	XPAY_FLOAT2(x2, a, y2);
	CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
	i += gridSize;
    } 
    
}

// performs the operation y[i] = x[i] + a*y[i]
void xpayCuda(ParitySpinor x, double a, ParitySpinor y) 
{
    checkSpinor(x,y);
    int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if (x.precision == QUDA_DOUBLE_PRECISION) {
	xpayKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, a, (double*)y.spinor, x.length);
    } else if (x.precision == QUDA_SINGLE_PRECISION) {
	xpayKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float)a, (float*)y.spinor, x.length);
    } else {
	int spinor_bytes = x.length*sizeof(short);
	cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
	xpayHKernel<<<dimGrid, dimBlock>>>((float)a, (short2*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
    }
}
void 
xpayFullCuda(FullSpinor x, double a, FullSpinor y) 
{
    xpayCuda(x.even, a, y.even);
    xpayCuda(x.odd, a, y.odd);
}

template <typename Float>
__global__ void axKernel(Float a, Float *x, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    x[i] *= a;
    i += gridSize;
  } 
}

template <typename Float>
__global__ void mxpyKernel(Float *x, Float *y, int len) 
{
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
	y[i] -= x[i];
	i += gridSize;
    }
}

__global__ void mxpyHKernel(short2 *yH, float *yN, int length) 
{
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < length) {
	RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
	RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
	MXPY_FLOAT2(x0, y0);
	MXPY_FLOAT2(x1, y1);
	MXPY_FLOAT2(x2, y2);
	CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
	i += gridSize;
    }    
}


// performs the operation y[i] -= x[i] (minus x plus y)
void mxpyCuda(ParitySpinor x, ParitySpinor y) {
  checkSpinor(x,y);
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
      mxpyKernel<<<dimGrid, dimBlock>>>((double*)x.spinor, (double*)y.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
      mxpyKernel<<<dimGrid, dimBlock>>>((float*)x.spinor, (float*)y.spinor, x.length);
  } else {
      int spinor_bytes = x.length*sizeof(short);
      cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes);
      cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);
      cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes);
      cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);
      mxpyHKernel<<<dimGrid, dimBlock>>>((short2*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
  }
}

void mxpyFullCuda(FullSpinor x, FullSpinor y) 
{
    mxpyCuda(x.even, y.even);
    mxpyCuda(x.odd, y.odd);
    
}

__global__ void axHKernel(float a, short2 *xH, float *xN, int length) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
      RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
      AX_FLOAT2(a, x0); AX_FLOAT2(a, x1); AX_FLOAT2(a, x2);
      CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, length);
      i += gridSize;
  } 
  
}

// performs the operation x[i] = a*x[i]
void axCuda(double a, ParitySpinor x) {
  int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
  dim3 dimBlock(REDUCE_THREADS, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  if (x.precision == QUDA_DOUBLE_PRECISION) {
    axKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, x.length);
  } else if (x.precision == QUDA_SINGLE_PRECISION) {
    axKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, x.length);
  } else {
    int spinor_bytes = x.length*sizeof(short);
    cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
    axHKernel<<<dimGrid, dimBlock>>>((float)a, (short2*)x.spinor, (float*)x.spinorNorm, x.length/spinorSiteSize);
  }
}



template <typename Float>
__global__ void axpyZpbxKernel(Float a, Float *x, Float *y, Float *z, Float b, int len) {
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < len) {
    Float x_i = x[i];
    y[i] += a*x_i;
    x[i] = z[i] + b*x_i;
    i += gridSize;
  }
}

__global__ void axpyZpbxHKernel(float a, float b, short2 *xH, float *xN, short2 *yH, float *yN, int length) {
  
  unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
  unsigned int gridSize = gridDim.x*blockDim.x;
  while (i < length) {
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
    RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, length);
    AXPY_FLOAT2(a, x0, y0);
    XPAY_FLOAT2(z0, b, x0);
    AXPY_FLOAT2(a, x1, y1);
    XPAY_FLOAT2(z1, b, x1);
    AXPY_FLOAT2(a, x2, y2);
    XPAY_FLOAT2(z2, b, x2);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, length);
    i += gridSize;
  }   
}


// performs the operations: {y[i] = a x[i] + y[i]; x[i] = z[i] + b x[i]}
void axpyZpbxCuda(double a, ParitySpinor x, ParitySpinor y, ParitySpinor z, double b) {
    checkSpinor(x,y);
    checkSpinor(x,z);
    int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if (x.precision == QUDA_DOUBLE_PRECISION) {
	axpyZpbxKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, (double*)z.spinor, b, x.length);
    } else if (x.precision == QUDA_SINGLE_PRECISION) {
	axpyZpbxKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, (float*)z.spinor, (float)b, x.length);
    } else {
	int spinor_bytes = x.length*sizeof(short);
	cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/3);    
	axpyZpbxHKernel<<<dimGrid, dimBlock>>>((float)a, (float)b, (short2*)x.spinor, (float*)x.spinorNorm,
					       (short2*)y.spinor, (float*)y.spinorNorm, z.length/spinorSiteSize);CUERR;
    }
}

template <typename Float>
__global__ void axpyBzpcxKernel(Float a, Float *x, Float *y, Float b, Float *z, Float c, int len) 
{
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < len) {
	Float x_i = x[i];
	y[i] += a*x_i;
	x[i] = b*z[i] + c*x_i;
	i += gridSize;
    }
}

__global__ void axpyBzpcxHKernel(float a, float b, float c, short2 *xH, float *xN, short2 *yH, float *yN, int length) 
{
    
    unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;
    while (i < length) {
	RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, length);
	RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, length);
	RECONSTRUCT_HALF_SPINOR(z, texHalf3, texNorm3, length);
	AXPY_FLOAT2(a, x0, y0);
	AXPBY_FLOAT2(b, z0, c, x0);
	AXPY_FLOAT2(a, x1, y1);
	AXPBY_FLOAT2(b, z1, c, x1);
	AXPY_FLOAT2(a, x2, y2);
	AXPBY_FLOAT2(b, z2, c, x2);
	CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, length);
	CONSTRUCT_HALF_SPINOR_FROM_SINGLE(xH, xN, x, length);
	i += gridSize;
    }   
}


// performs the operations: {y[i] = a x[i] + y[i]; x[i] = b z[i] + c x[i]}
void axpyBzpcxCuda(double a, ParitySpinor x, ParitySpinor y, double b, ParitySpinor z, double c) {
    checkSpinor(x,y);
    checkSpinor(x,z);
    int blocks = min(REDUCE_MAX_BLOCKS, max(x.length/REDUCE_THREADS, 1));
    dim3 dimBlock(REDUCE_THREADS, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    if (x.precision == QUDA_DOUBLE_PRECISION) {
	axpyBzpcxKernel<<<dimGrid, dimBlock>>>(a, (double*)x.spinor, (double*)y.spinor, b, (double*)z.spinor, c, x.length);
    } else if (x.precision == QUDA_SINGLE_PRECISION) {
	axpyBzpcxKernel<<<dimGrid, dimBlock>>>((float)a, (float*)x.spinor, (float*)y.spinor, (float)b, (float*)z.spinor, (float)c, x.length);
    } else {
	int spinor_bytes = x.length*sizeof(short);
	cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf3, z.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm3, z.spinorNorm, spinor_bytes/3);    
	axpyBzpcxHKernel<<<dimGrid, dimBlock>>>((float)a, (float)b, (float)c, (short2*)x.spinor, (float*)x.spinorNorm,
						(short2*)y.spinor, (float*)y.spinorNorm, z.length/spinorSiteSize);CUERR;
    }
}

void
axpyZpbxFullCuda(double a, FullSpinor x, FullSpinor y, FullSpinor z, double b) 
{
    axpyZpbxCuda(a, x.even, y.even, z.even, b);
    axpyZpbxCuda(a, x.odd, y.odd, z.odd, b);    
}


// Computes c = a + b in "double single" precision.
__device__ void dsadd(QudaSumFloat &c0, QudaSumFloat &c1, const QudaSumFloat a0, 
		      const QudaSumFloat a1, const float b0, const float b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0 + b0;
  QudaSumFloat e = t1 - a0;
  QudaSumFloat t2 = ((b0 - e) + (a0 - (t1 - e))) + a1 + b1;
  // The result is t1 + t2, after normalization.
  c0 = e = t1 + t2;
  c1 = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (complex version)
__device__ void zcadd(QudaSumComplex &c0, QudaSumComplex &c1, const QudaSumComplex a0, 
		      const QudaSumComplex a1, const QudaSumComplex b0, const QudaSumComplex b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
}

// Computes c = a + b in "double single" precision (float3 version)
__device__ void dsadd3(QudaSumFloat3 &c0, QudaSumFloat3 &c1, const QudaSumFloat3 a0, 
		       const QudaSumFloat3 a1, const QudaSumFloat3 b0, const QudaSumFloat3 b1) {
  // Compute dsa + dsb using Knuth's trick.
  QudaSumFloat t1 = a0.x + b0.x;
  QudaSumFloat e = t1 - a0.x;
  QudaSumFloat t2 = ((b0.x - e) + (a0.x - (t1 - e))) + a1.x + b1.x;
  // The result is t1 + t2, after normalization.
  c0.x = e = t1 + t2;
  c1.x = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.y + b0.y;
  e = t1 - a0.y;
  t2 = ((b0.y - e) + (a0.y - (t1 - e))) + a1.y + b1.y;
  // The result is t1 + t2, after normalization.
  c0.y = e = t1 + t2;
  c1.y = t2 - (e - t1);
  
  // Compute dsa + dsb using Knuth's trick.
  t1 = a0.z + b0.z;
  e = t1 - a0.z;
  t2 = ((b0.z - e) + (a0.z - (t1 - e))) + a1.z + b1.z;
  // The result is t1 + t2, after normalization.
  c0.z = e = t1 + t2;
  c1.z = t2 - (e - t1);
}


//
// double normCuda(float *a, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) normF##suffix
#define REDUCE_TYPES Float *a
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) (a[i]*a[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

//
// double normHCuda(char *, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) normH##suffix
#define REDUCE_TYPES Float *a // dummy type
#define REDUCE_PARAMS a
#define REDUCE_AUXILIARY(i)						\
    RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, n);		\
    REAL_DOT_FLOAT2(norm0, I0, I0);					\
    REAL_DOT_FLOAT2(norm1, I1, I1);					\
    REAL_DOT_FLOAT2(norm2, I2, I2);					\
    norm0 += norm1; norm0 += norm2; 
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION


double normCuda(ParitySpinor a) 
{
    if (a.precision == QUDA_DOUBLE_PRECISION) {
	double ret = normFCuda((double*)a.spinor, a.length);
	return ret;
    } else if (a.precision == QUDA_SINGLE_PRECISION) {
	double ret =  normFCuda((float*)a.spinor, a.length);
	return ret;
    } else {
	int spinor_bytes = a.length*sizeof(short);
	cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/3);    
	return normHCuda((char*)0, a.length/spinorSiteSize);
    }
}


double normFullCuda(FullSpinor a) 
{
    return normCuda(a.even)+ normCuda(a.odd);
}


//
// double relativeNormCuda(float *p, float* q, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) relativeNormF##suffix
#define REDUCE_TYPES Float *p, Float* q
#define REDUCE_PARAMS p, q
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i)						\
    ((p[6*i ]*p[6*i  ]  +  p[6*i+1]*p[6*i+1]+ p[6*i+2]*p[6*i+2] +	\
      p[6*i+3]*p[6*i+3] +  p[6*i+4]*p[6*i+4]+ p[6*i+5]*p[6*i+5])/ 	\
     (q[6*i ]*q[6*i  ]  +  q[6*i+1]*q[6*i+1]+ q[6*i+2]*q[6*i+2] +	\
      q[6*i+3]*q[6*i+3] +  q[6*i+4]*q[6*i+4]+ q[6*i+5]*q[6*i+5]))	   


#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

//
// double relativeNormHCuda(char *,  int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) relativeNormH##suffix
#define REDUCE_TYPES Float *p // dummy type
#define REDUCE_PARAMS p
#define REDUCE_AUXILIARY(i)					\
    RECONSTRUCT_HALF_SPINOR(I, texHalf1, texNorm1, n);			\
    REAL_DOT_FLOAT2(pnorm0, I0, I0);					\
    REAL_DOT_FLOAT2(pnorm1, I1, I1);					\
    REAL_DOT_FLOAT2(pnorm2, I2, I2);					\
    pnorm0 += pnorm1; pnorm0 += pnorm2;					\
    RECONSTRUCT_HALF_SPINOR(J, texHalf2, texNorm2, n);			\
    REAL_DOT_FLOAT2(qnorm0, J0, J0);					\
    REAL_DOT_FLOAT2(qnorm1, J1, J1);					\
    REAL_DOT_FLOAT2(qnorm2, J2, J2);					\
    qnorm0 += qnorm1; qnorm0 += qnorm2;					
#define REDUCE_OPERATION(i) (pnorm0/qnorm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double relativeNormCuda(ParitySpinor p, ParitySpinor q)
{
    
    if ( (p.length != q.length)
	 ||(p.precision != q.precision)){
	fprintf(stderr, "ERROR: in function %s, Spinor p and q are not the same, type\n", __FUNCTION__ );
	exit(1);
    }
    if (p.precision == QUDA_DOUBLE_PRECISION) {
	double residue = relativeNormFCuda((double*)p.spinor, (double*)q.spinor, p.length/spinorSiteSize);
	double ret = sqrt(residue/p.volume);
	return ret;
    } else if (p.precision == QUDA_SINGLE_PRECISION) {
	double residue =  relativeNormFCuda((float*)p.spinor, (float*)q.spinor,  p.length/spinorSiteSize);
	double ret = sqrt(residue/p.volume);
	return ret;
    } else {
	int spinor_bytes = p.length*sizeof(short);
	cudaBindTexture(0, texHalf1, p.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, p.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf2, q.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm2, q.spinorNorm, spinor_bytes/3);    
	double residue = relativeNormHCuda((char*)0, p.length/spinorSiteSize);
	double ret = sqrt(residue /p.volume);
	return ret;
    }
}



//
// double reDotProductFCuda(float *a, float *b, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProductF##suffix
#define REDUCE_TYPES Float *a, Float *b
#define REDUCE_PARAMS a, b
#define REDUCE_AUXILIARY(i)
#define REDUCE_OPERATION(i) (a[i]*b[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

//
// double reDotProductHCuda(float *a, float *b, int n) {}
//
template <typename Float>
#define REDUCE_FUNC_NAME(suffix) reDotProductH##suffix
#define REDUCE_TYPES Float *a, Float *b
#define REDUCE_PARAMS a, b
#define REDUCE_AUXILIARY(i)						\
    RECONSTRUCT_HALF_SPINOR(aH, texHalf1, texNorm1, n);		\
    RECONSTRUCT_HALF_SPINOR(bH, texHalf2, texNorm2, n);		\
    REAL_DOT_FLOAT2(rdot0, aH0, bH0);					\
    REAL_DOT_FLOAT2(rdot1, aH1, bH1);					\
    REAL_DOT_FLOAT2(rdot2, aH2, bH2);					\
    rdot0 += rdot1; rdot0 += rdot2;					
#define REDUCE_OPERATION(i) (rdot0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double reDotProductCuda(ParitySpinor a, ParitySpinor b) {
  checkSpinor(a, b);
  if (a.precision == QUDA_DOUBLE_PRECISION) {
    return reDotProductFCuda((double*)a.spinor, (double*)b.spinor, a.length);
  } else if (a.precision == QUDA_SINGLE_PRECISION) {
    return reDotProductFCuda((float*)a.spinor, (float*)b.spinor, a.length);
  } else {
    int spinor_bytes = a.length*sizeof(short);
    cudaBindTexture(0, texHalf1, a.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm1, a.spinorNorm, spinor_bytes/3);    
    cudaBindTexture(0, texHalf2, b.spinor, spinor_bytes); 
    cudaBindTexture(0, texNorm2, b.spinorNorm, spinor_bytes/3);    
    return reDotProductHCuda((char*)0, (char*)0, a.length/spinorSiteSize);
  }
}

double
reDotProductFullCuda(FullSpinor a, FullSpinor b) 
{
    return  reDotProductCuda(a.even, b.even) + reDotProductCuda(a.odd, b.odd);
}

//
// double axpyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = a*x[i] + y[i]
// Second returns the norm of y
//

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNormF##suffix
#define REDUCE_TYPES Float a, Float *x, Float *y
#define REDUCE_PARAMS a, x, y
#define REDUCE_AUXILIARY(i) y[i] = a*x[i] + y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) axpyNormH##suffix
#define REDUCE_TYPES Float a, short2 *yH, float *yN
#define REDUCE_PARAMS a, yH, yN
#define REDUCE_AUXILIARY(i)						\
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);		\
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);		\
    AXPY_FLOAT2(a, x0, y0);						\
    REAL_DOT_FLOAT2(norm0, y0, y0);					\
    AXPY_FLOAT2(a, x1, y1);						\
    REAL_DOT_FLOAT2(norm1, y1, y1);					\
    AXPY_FLOAT2(a, x2, y2);						\
    REAL_DOT_FLOAT2(norm2, y2, y2);					\
    norm0 += norm1; norm0 += norm2;					\
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, n);
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double axpyNormCuda(double a, ParitySpinor x, ParitySpinor y) {
    checkSpinor(x,y);
    if (x.precision == QUDA_DOUBLE_PRECISION) {
	return axpyNormFCuda(a, (double*)x.spinor, (double*)y.spinor, x.length);
    } else if (x.precision == QUDA_SINGLE_PRECISION) {
	return axpyNormFCuda((float)a, (float*)x.spinor, (float*)y.spinor, x.length);
    } else {
	int spinor_bytes = x.length*sizeof(short);
	cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
	return axpyNormHCuda((float)a, (short2*)y.spinor, (float*)y.spinorNorm, x.length/spinorSiteSize);
    }
}

double 
axpyNormFullCuda(double a, FullSpinor x, FullSpinor y)
{
    double ret;
    
    ret = axpyNormCuda(a, x.even, y.even);
    ret += axpyNormCuda(a, x.odd, y.odd);
    return ret;
}

//
// double xmyNormCuda(float a, float *x, float *y, n){}
//
// First performs the operation y[i] = x[i] - y[i]
// Second returns the norm of y
//

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) xmyNormF##suffix
#define REDUCE_TYPES Float *x, Float *y
#define REDUCE_PARAMS x, y
#define REDUCE_AUXILIARY(i) y[i] = x[i] - y[i]
#define REDUCE_OPERATION(i) (y[i]*y[i])
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

template <typename Float>
#define REDUCE_FUNC_NAME(suffix) xmyNormH##suffix
#define REDUCE_TYPES Float *d1, Float *d2, short2 *yH, float *yN
#define REDUCE_PARAMS d1, d2, yH, yN
#define REDUCE_AUXILIARY(i)						\
    RECONSTRUCT_HALF_SPINOR(x, texHalf1, texNorm1, n);			\
    RECONSTRUCT_HALF_SPINOR(y, texHalf2, texNorm2, n);		\
    XMY_FLOAT2(x0, y0);							\
    REAL_DOT_FLOAT2(norm0, y0, y0);					\
    XMY_FLOAT2(x1, y1);							\
    REAL_DOT_FLOAT2(norm1, y1, y1);					\
    XMY_FLOAT2(x2, y2);							\
    REAL_DOT_FLOAT2(norm2, y2, y2);					\
    norm0 += norm1; norm0 += norm2;					\
    CONSTRUCT_HALF_SPINOR_FROM_SINGLE(yH, yN, y, n);
#define REDUCE_OPERATION(i) (norm0)
#include "reduce_core.h"
#undef REDUCE_FUNC_NAME
#undef REDUCE_TYPES
#undef REDUCE_PARAMS
#undef REDUCE_AUXILIARY
#undef REDUCE_OPERATION

double xmyNormCuda(ParitySpinor x, ParitySpinor y) 
{
    checkSpinor(x,y);
    if (x.precision == QUDA_DOUBLE_PRECISION) {
	return xmyNormFCuda((double*)x.spinor, (double*)y.spinor, x.length);
    } else if (x.precision == QUDA_SINGLE_PRECISION) {
	return xmyNormFCuda((float*)x.spinor, (float*)y.spinor, x.length);
    } else { 
	int spinor_bytes = x.length*sizeof(short);
	cudaBindTexture(0, texHalf1, x.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm1, x.spinorNorm, spinor_bytes/3);    
	cudaBindTexture(0, texHalf2, y.spinor, spinor_bytes); 
	cudaBindTexture(0, texNorm2, y.spinorNorm, spinor_bytes/3);    
	return xmyNormHCuda((char*)0, (char*)0, (short2*)y.spinor, (float*)y.spinorNorm, y.length/spinorSiteSize);
    }
}

double 
xmyNormFullCuda(FullSpinor x, FullSpinor y) 
{
    return xmyNormCuda(x.even, y.even) + xmyNormCuda(x.odd, y.odd);
}


double cpuDouble(float *data, int size) {
  double sum = 0;
  for (int i = 0; i < size; i++) sum += data[i];
  return sum;
}

/*
void blasTest() {
  int n = 3*1<<24;
  float *h_data = (float *)malloc(n*sizeof(float));
  float *d_data;
  if (cudaMalloc((void **)&d_data,  n*sizeof(float))) {
    printf("Error allocating d_data\n");
    exit(0);
  }
  
  for (int i = 0; i < n; i++) {
    h_data[i] = rand()/(float)RAND_MAX - 0.5; // n-1.0-i;
  }
  
  cudaMemcpy(d_data, h_data, n*sizeof(float), cudaMemcpyHostToDevice);
  
  cudaThreadSynchronize();
  stopwatchStart();
  int LOOPS = 20;
  for (int i = 0; i < LOOPS; i++) {
    sumCuda(d_data, n);
  }
  cudaThreadSynchronize();
  float secs = stopwatchReadSeconds();
  
  printf("%f GiB/s\n", 1.e-9*n*sizeof(float)*LOOPS / secs);
  printf("Device: %f MiB\n", (float)n*sizeof(float) / (1 << 20));
  printf("Shared: %f KiB\n", (float)REDUCE_THREADS*sizeof(float) / (1 << 10));
  
  float correctDouble = cpuDouble(h_data, n);
  printf("Total: %f\n", correctDouble);
  printf("CUDA db: %f\n", fabs(correctDouble-sumCuda(d_data, n)));
  
  cudaFree(d_data) ;
  free(h_data);
}
*/
/*
void axpbyTest() {
    int n = 3 * 1 << 20;
    float *h_x = (float *)malloc(n*sizeof(float));
    float *h_y = (float *)malloc(n*sizeof(float));
    float *h_res = (float *)malloc(n*sizeof(float));
    
    float *d_x, *d_y;
    if (cudaMalloc((void **)&d_x,  n*sizeof(float))) {
      printf("Error allocating d_x\n");
      exit(0);
    }
    if (cudaMalloc((void **)&d_y,  n*sizeof(float))) {
      printf("Error allocating d_y\n");
      exit(0);
    }
    
    for (int i = 0; i < n; i++) {
        h_x[i] = 1;
        h_y[i] = 2;
    }
    
    cudaMemcpy(d_x, h_x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n*sizeof(float), cudaMemcpyHostToDevice);
    
    axpbyCuda(4, d_x, 3, d_y, n/2);
    
    cudaMemcpy( h_res, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float expect = (i < n/2) ? 4*h_x[i] + 3*h_y[i] : h_y[i];
        if (h_res[i] != expect)
            printf("FAILED %d : %f != %f\n", i, h_res[i], h_y[i]);
    }
    
    cudaFree(d_y);
    cudaFree(d_x);
    free(h_x);
    free(h_y);
}
*/
