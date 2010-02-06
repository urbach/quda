// spinor_quda.cpp
// Ver. 09.10.a

#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <spinor_quda.h>

#include <xmmintrin.h>


// Pinned memory for cpu-gpu memory copying
void *packedSpinor1 = 0;
void *packedSpinor2 = 0;

// This gets called, for instance, in dslash_test.c, prior to any
// calls that perform packing and transfer to device mem.
//ok
ParitySpinor allocateParitySpinor(int geometric_length, Precision precision) {
  ParitySpinor ret;

  ret.precision = precision;
  ret.length = geometric_length*spinorSiteSize;

  if (precision == QUDA_DOUBLE_PRECISION) {
    int spinor_bytes = ret.length*sizeof(double);
    if (cudaMalloc((void**)&ret.spinor, spinor_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinor\n");
      exit(0);
    }
  } else if (precision == QUDA_SINGLE_PRECISION) {
    int spinor_bytes = ret.length*sizeof(float);
    if (cudaMalloc((void**)&ret.spinor, spinor_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinor\n");
      exit(0);
    }
  } else if (precision == QUDA_HALF_PRECISION) {
    int spinor_bytes = ret.length*sizeof(float)/2;
    if (cudaMalloc((void**)&ret.spinor, spinor_bytes) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinor\n");
      exit(0);
    }
    if (cudaMalloc((void**)&ret.spinorNorm, spinor_bytes/12) == cudaErrorMemoryAllocation) {
      printf("Error allocating spinorNorm\n");
      exit(0);
    }
  }

  return ret;
}


FullSpinor allocateSpinorField(int length, Precision precision) {
  FullSpinor ret;
  ret.even = allocateParitySpinor(length/2, precision);
  ret.odd = allocateParitySpinor(length/2, precision);
  return ret;
}



void freeParitySpinor(ParitySpinor spinor) {

  cudaFree(spinor.spinor);
  if (spinor.precision == QUDA_HALF_PRECISION) cudaFree(spinor.spinorNorm);

  spinor.spinor = NULL;
  spinor.spinorNorm = NULL;
}


void freeSpinorField(FullSpinor spinor) {
  freeParitySpinor(spinor.even);
  freeParitySpinor(spinor.odd);
}

void freeSpinorBuffer() {
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedSpinor1);
#else
  free(packedSpinor1);
#endif
  packedSpinor1 = NULL;
}

template <typename Float>
inline void packSpinorVector(float4* a, Float *b) {
  Float K = 1.0 / 2.0;

  a[0*Nh_5d].x = K*(b[1*6+0*2+0]+b[3*6+0*2+0]);
  a[0*Nh_5d].y = K*(b[1*6+0*2+1]+b[3*6+0*2+1]);
  a[0*Nh_5d].z = K*(b[1*6+1*2+0]+b[3*6+1*2+0]);
  a[0*Nh_5d].w = K*(b[1*6+1*2+1]+b[3*6+1*2+1]);
  
  a[1*Nh_5d].x = K*(b[1*6+2*2+0]+b[3*6+2*2+0]);
  a[1*Nh_5d].y = K*(b[1*6+2*2+1]+b[3*6+2*2+1]);
  a[1*Nh_5d].z = -K*(b[2*6+0*2+0]+b[0*6+0*2+0]);
  a[1*Nh_5d].w = -K*(b[2*6+0*2+1]+b[0*6+0*2+1]);
  
  a[2*Nh_5d].x = -K*(b[0*6+1*2+0]+b[2*6+1*2+0]);
  a[2*Nh_5d].y = -K*(b[0*6+1*2+1]+b[2*6+1*2+1]);
  a[2*Nh_5d].z = -K*(b[0*6+2*2+0]+b[2*6+2*2+0]);
  a[2*Nh_5d].w = -K*(b[0*6+2*2+1]+b[2*6+2*2+1]);

  a[3*Nh_5d].x = K*(b[1*6+0*2+0]-b[3*6+0*2+0]);
  a[3*Nh_5d].y = K*(b[1*6+0*2+1]-b[3*6+0*2+1]);
  a[3*Nh_5d].z = K*(b[1*6+1*2+0]-b[3*6+1*2+0]);
  a[3*Nh_5d].w = K*(b[1*6+1*2+1]-b[3*6+1*2+1]);

  a[4*Nh_5d].x = K*(b[1*6+2*2+0]-b[3*6+2*2+0]);
  a[4*Nh_5d].y = K*(b[1*6+2*2+1]-b[3*6+2*2+1]);
  a[4*Nh_5d].z = K*(b[2*6+0*2+0]-b[0*6+0*2+0]);
  a[4*Nh_5d].w = K*(b[2*6+0*2+1]-b[0*6+0*2+1]);

  a[5*Nh_5d].x = K*(b[2*6+1*2+0]-b[0*6+1*2+0]);
  a[5*Nh_5d].y = K*(b[2*6+1*2+1]-b[0*6+1*2+1]);
  a[5*Nh_5d].z = K*(b[2*6+2*2+0]-b[0*6+2*2+0]);
  a[5*Nh_5d].w = K*(b[2*6+2*2+1]-b[0*6+2*2+1]);
}

template <typename Float>
inline void packQDPSpinorVector(float4* a, Float *b) {
  Float K = 1.0 / 2.0;

  a[0*Nh_5d].x = K*(b[(0*4+1)*2+0]+b[(0*4+3)*2+0]);
  a[0*Nh_5d].y = K*(b[(0*4+1)*2+1]+b[(0*4+3)*2+1]);
  a[0*Nh_5d].z = K*(b[(1*4+1)*2+0]+b[(1*4+3)*2+0]);
  a[0*Nh_5d].w = K*(b[(1*4+1)*2+1]+b[(1*4+3)*2+1]);

  a[1*Nh_5d].x = K*(b[(2*4+1)*2+0]+b[(2*4+3)*2+0]);
  a[1*Nh_5d].y = K*(b[(2*4+1)*2+1]+b[(2*4+3)*2+1]);
  a[1*Nh_5d].z = -K*(b[(0*4+0)*2+0]+b[(0*4+2)*2+0]);
  a[1*Nh_5d].w = -K*(b[(0*4+0)*2+1]+b[(0*4+2)*2+1]);

  a[2*Nh_5d].x = -K*(b[(1*4+0)*2+0]+b[(1*4+2)*2+0]);
  a[2*Nh_5d].y = -K*(b[(1*4+0)*2+1]+b[(1*4+2)*2+1]);
  a[2*Nh_5d].z = -K*(b[(2*4+0)*2+0]+b[(2*4+2)*2+0]);
  a[2*Nh_5d].w = -K*(b[(2*4+0)*2+1]+b[(2*4+2)*2+1]);

  a[3*Nh_5d].x = K*(b[(0*4+1)*2+0]+b[(0*4+3)*2+0]);
  a[3*Nh_5d].y = K*(b[(0*4+1)*2+1]+b[(0*4+3)*2+1]);
  a[3*Nh_5d].z = K*(b[(1*4+1)*2+0]+b[(1*4+3)*2+0]);
  a[3*Nh_5d].w = K*(b[(1*4+1)*2+1]+b[(1*4+3)*2+1]);

  a[4*Nh_5d].x = K*(b[(2*4+1)*2+0]+b[(2*4+3)*2+0]);
  a[4*Nh_5d].y = K*(b[(2*4+1)*2+1]+b[(2*4+3)*2+1]);
  a[4*Nh_5d].z = K*(b[(0*4+2)*2+0]+b[(0*4+0)*2+0]);
  a[4*Nh_5d].w = K*(b[(0*4+2)*2+1]+b[(0*4+0)*2+1]);

  a[5*Nh_5d].x = K*(b[(1*4+2)*2+0]+b[(1*4+0)*2+0]);
  a[5*Nh_5d].y = K*(b[(1*4+2)*2+1]+b[(1*4+0)*2+1]);
  a[5*Nh_5d].z = K*(b[(2*4+2)*2+0]+b[(2*4+0)*2+0]);
  a[5*Nh_5d].w = K*(b[(2*4+2)*2+1]+b[(2*4+0)*2+1]);
}

template <typename Float>
inline void packSpinorVector(double2* a, Float *b) {
  Float K = 1.0 / 2.0;

  for (int c=0; c<3; c++) {
    a[c*Nh_5d].x = K*(b[1*6+c*2+0]+b[3*6+c*2+0]);
    a[c*Nh_5d].y = K*(b[1*6+c*2+1]+b[3*6+c*2+1]);

    a[(3+c)*Nh_5d].x = -K*(b[0*6+c*2+0]+b[2*6+c*2+0]);
    a[(3+c)*Nh_5d].y = -K*(b[0*6+c*2+1]+b[2*6+c*2+1]);

    a[(6+c)*Nh_5d].x = K*(b[1*6+c*2+0]-b[3*6+c*2+0]);
    a[(6+c)*Nh_5d].y = K*(b[1*6+c*2+1]-b[3*6+c*2+1]);

    a[(9+c)*Nh_5d].x = K*(b[2*6+c*2+0]-b[0*6+c*2+0]);
    a[(9+c)*Nh_5d].y = K*(b[2*6+c*2+1]-b[0*6+c*2+1]);
  }

}

template <typename Float>
inline void packQDPSpinorVector(double2* a, Float *b) {
  Float K = 1.0 / 2.0;

  for (int c=0; c<3; c++) {
    a[c*Nh_5d].x = K*(b[(c*4+1)*2+0]+b[(c*4+3)*2+0]);
    a[c*Nh_5d].y = K*(b[(c*4+1)*2+1]+b[(c*4+3)*2+1]);

    a[(3+c)*Nh_5d].x = -K*(b[(c*4+0)*2+0]+b[(c*4+2)*2+0]);
    a[(3+c)*Nh_5d].y = -K*(b[(c*4+0)*2+1]+b[(c*4+2)*2+1]);

    a[(6+c)*Nh_5d].x = K*(b[(c*4+1)*2+0]-b[(c*4+3)*2+0]);
    a[(6+c)*Nh_5d].y = K*(b[(c*4+1)*2+1]-b[(c*4+3)*2+1]);

    a[(9+c)*Nh_5d].x = K*(b[(c*4+2)*2+0]-b[(c*4+0)*2+0]);
    a[(9+c)*Nh_5d].y = K*(b[(c*4+2)*2+1]-b[(c*4+0)*2+1]);
  }

}

template <typename Float>
inline void unpackSpinorVector(Float *a, float4 *b) {
  Float K = 1.0;

  a[0*6+0*2+0] = -K*(b[Nh_5d].z+b[4*Nh_5d].z);
  a[0*6+0*2+1] = -K*(b[Nh_5d].w+b[4*Nh_5d].w);
  a[0*6+1*2+0] = -K*(b[2*Nh_5d].x+b[5*Nh_5d].x);
  a[0*6+1*2+1] = -K*(b[2*Nh_5d].y+b[5*Nh_5d].y);
  a[0*6+2*2+0] = -K*(b[2*Nh_5d].z+b[5*Nh_5d].z);
  a[0*6+2*2+1] = -K*(b[2*Nh_5d].w+b[5*Nh_5d].w);
  
  a[1*6+0*2+0] = K*(b[0].x+b[3*Nh_5d].x);
  a[1*6+0*2+1] = K*(b[0].y+b[3*Nh_5d].y);
  a[1*6+1*2+0] = K*(b[0].z+b[3*Nh_5d].z);
  a[1*6+1*2+1] = K*(b[0].w+b[3*Nh_5d].w);  
  a[1*6+2*2+0] = K*(b[Nh_5d].x+b[4*Nh_5d].x);
  a[1*6+2*2+1] = K*(b[Nh_5d].y+b[4*Nh_5d].y);
  
  a[2*6+0*2+0] = -K*(b[Nh_5d].z-b[4*Nh_5d].z);
  a[2*6+0*2+1] = -K*(b[Nh_5d].w-b[4*Nh_5d].w);
  a[2*6+1*2+0] = -K*(b[2*Nh_5d].x-b[5*Nh_5d].x);
  a[2*6+1*2+1] = -K*(b[2*Nh_5d].y-b[5*Nh_5d].y);
  a[2*6+2*2+0] = -K*(b[2*Nh_5d].z-b[5*Nh_5d].z);
  a[2*6+2*2+1] = -K*(b[2*Nh_5d].w-b[5*Nh_5d].w);
  
  a[3*6+0*2+0] = -K*(b[3*Nh_5d].x-b[0].x);
  a[3*6+0*2+1] = -K*(b[3*Nh_5d].y-b[0].y);
  a[3*6+1*2+0] = -K*(b[3*Nh_5d].z-b[0].z);
  a[3*6+1*2+1] = -K*(b[3*Nh_5d].w-b[0].w);
  a[3*6+2*2+0] = -K*(b[4*Nh_5d].x-b[Nh_5d].x);
  a[3*6+2*2+1] = -K*(b[4*Nh_5d].y-b[Nh_5d].y);
}

template <typename Float>
inline void unpackQDPSpinorVector(Float *a, float4 *b) {
  Float K = 1.0;

  a[(0*4+0)*2+0] = -K*(b[Nh_5d].z+b[4*Nh_5d].z);
  a[(0*4+0)*2+1] = -K*(b[Nh_5d].w+b[4*Nh_5d].w);
  a[(1*4+0)*2+0] = -K*(b[2*Nh_5d].x+b[5*Nh_5d].x);
  a[(1*4+0)*2+1] = -K*(b[2*Nh_5d].y+b[5*Nh_5d].y);
  a[(2*4+0)*2+0] = -K*(b[2*Nh_5d].z+b[5*Nh_5d].z);
  a[(2*4+0)*2+1] = -K*(b[2*Nh_5d].w+b[5*Nh_5d].w);
  
  a[(0*4+1)*2+0] = K*(b[0].x+b[3*Nh_5d].x);
  a[(0*4+1)*2+1] = K*(b[0].y+b[3*Nh_5d].y);
  a[(1*4+1)*2+0] = K*(b[0].z+b[3*Nh_5d].z);
  a[(1*4+1)*2+1] = K*(b[0].w+b[3*Nh_5d].w);  
  a[(2*4+1)*2+0] = K*(b[Nh_5d].x+b[4*Nh_5d].x);
  a[(2*4+1)*2+1] = K*(b[Nh_5d].y+b[4*Nh_5d].y);
  
  a[(0*4+2)*2+0] = -K*(b[Nh_5d].z-b[4*Nh_5d].z);
  a[(0*4+2)*2+1] = -K*(b[Nh_5d].w-b[4*Nh_5d].w);
  a[(1*4+2)*2+0] = -K*(b[2*Nh_5d].x-b[5*Nh_5d].x);
  a[(1*4+2)*2+1] = -K*(b[2*Nh_5d].y-b[5*Nh_5d].y);
  a[(2*4+2)*2+0] = -K*(b[2*Nh_5d].z-b[5*Nh_5d].z);
  a[(2*4+2)*2+1] = -K*(b[2*Nh_5d].w-b[5*Nh_5d].w);
  
  a[(0*4+3)*2+0] = -K*(b[3*Nh_5d].x-b[0].x);
  a[(0*4+3)*2+1] = -K*(b[3*Nh_5d].y-b[0].y);
  a[(1*4+3)*2+0] = -K*(b[3*Nh_5d].z-b[0].z);
  a[(1*4+3)*2+1] = -K*(b[3*Nh_5d].w-b[0].w);
  a[(2*4+3)*2+0] = -K*(b[4*Nh_5d].x-b[Nh_5d].x);
  a[(2*4+3)*2+1] = -K*(b[4*Nh_5d].y-b[Nh_5d].y);
}

template <typename Float>
inline void unpackSpinorVector(Float *a, double2 *b) {
  Float K = 1.0;

  for (int c=0; c<3; c++) {
    a[0*6+c*2+0] = -K*(b[(3+c)*Nh_5d].x+b[(9+c)*Nh_5d].x);
    a[0*6+c*2+1] = -K*(b[(3+c)*Nh_5d].y+b[(9+c)*Nh_5d].y);

    a[1*6+c*2+0] = K*(b[c*Nh_5d].x+b[(6+c)*Nh_5d].x);
    a[1*6+c*2+1] = K*(b[c*Nh_5d].y+b[(6+c)*Nh_5d].y);

    a[2*6+c*2+0] = -K*(b[(3+c)*Nh_5d].x-b[(9+c)*Nh_5d].x);
    a[2*6+c*2+1] = -K*(b[(3+c)*Nh_5d].y-b[(9+c)*Nh_5d].y);
    
    a[3*6+c*2+0] = -K*(b[(6+c)*Nh_5d].x-b[c*Nh_5d].x);
    a[3*6+c*2+1] = -K*(b[(6+c)*Nh_5d].y-b[c*Nh_5d].y);
  }

}

template <typename Float>
inline void unpackQDPSpinorVector(Float *a, double2 *b) {
  Float K = 1.0;

  for (int c=0; c<3; c++) {
    a[(c*4+0)*2+0] = -K*(b[(3+c)*Nh_5d].x+b[(9+c)*Nh_5d].x);
    a[(c*4+0)*2+1] = -K*(b[(3+c)*Nh_5d].y+b[(9+c)*Nh_5d].y);

    a[(c*4+1)*2+0] = K*(b[c*Nh_5d].x+b[(6+c)*Nh_5d].x);
    a[(c*4+1)*2+1] = K*(b[c*Nh_5d].y+b[(6+c)*Nh_5d].y);

    a[(c*4+2)*2+0] = -K*(b[(3+c)*Nh_5d].x-b[(9+c)*Nh_5d].x);
    a[(c*4+2)*2+1] = -K*(b[(3+c)*Nh_5d].y-b[(9+c)*Nh_5d].y);
    
    a[(c*4+3)*2+0] = -K*(b[(6+c)*Nh_5d].x-b[c*Nh_5d].x);
    a[(c*4+3)*2+1] = -K*(b[(6+c)*Nh_5d].y-b[c*Nh_5d].y);
  }

}

// Standard spinor packing, colour inside spin
template <typename Float, typename FloatN>
void packParitySpinor(FloatN *res, Float *spinor) {
  for (int i = 0; i < Nh_5d; i++) {
    packSpinorVector(res+i, spinor+24*i);
  }
}

template <typename Float, typename FloatN>
void packFullSpinor(FloatN *even, FloatN *odd, Float *spinor) {

  for (int i=0; i<Nh_5d; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      packSpinorVector(even+i, spinor+24*k);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      packSpinorVector(odd+i, spinor+24*k);
    }
  }

}

template <typename Float, typename FloatN>
void unpackFullSpinor(Float *res, FloatN *even, FloatN *odd) {

  for (int i=0; i<Nh_5d; i++) {

    int boundaryCrossings = i/L1h + i/(L2*L1h) + i/(L3*L2*L1h);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      unpackSpinorVector(res+24*k, even+i);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      unpackSpinorVector(res+24*k, odd+i);
    }
  }

}


template <typename Float, typename FloatN>
void unpackParitySpinor(Float *res, FloatN *spinorPacked) {

  for (int i = 0; i < Nh_5d; i++) {
    unpackSpinorVector(res+i*24, spinorPacked+i);
  }

}

// QDP spinor packing, spin inside colour
template <typename Float, typename FloatN>
void packQDPParitySpinor(FloatN *res, Float *spinor) {
  for (int i = 0; i < Nh_5d; i++) {
    packQDPSpinorVector(res+i, spinor+i*24);
  }
}

// QDP spinor packing, spin inside colour
template <typename Float, typename FloatN>
void unpackQDPParitySpinor(Float *res, FloatN *spinor) {
  for (int i = 0; i < Nh_5d; i++) {
    unpackQDPSpinorVector(res+i*24, spinor+i);
  }
}

void loadParitySpinor(ParitySpinor ret, void *spinor, Precision cpu_prec, 
		      DiracFieldOrder dirac_order) {

  if (ret.precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    printf("Error, cannot have CUDA double precision without double CPU precision\n");
    exit(-1);
  }

  if (ret.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (ret.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh_5d*spinorSiteSize*sizeof(double);
    else spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float);

#ifndef __DEVICE_EMULATION__
    if (!packedSpinor1) cudaMallocHost(&packedSpinor1, spinor_bytes);
    // We're going here.
    printf("cudaMallocHost called\n");
#else
    if (!packedSpinor1) packedSpinor1 = malloc(spinor_bytes);
#endif
    
    if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
      if (ret.precision == QUDA_DOUBLE_PRECISION) {
	      packParitySpinor((double2*)packedSpinor1, (double*)spinor);
        // We're going here.
        printf("double2 packing performed\n");
      } else {
      	if (cpu_prec == QUDA_DOUBLE_PRECISION) packParitySpinor((float4*)packedSpinor1, (double*)spinor);
      	else packParitySpinor((float4*)packedSpinor1, (float*)spinor);
      }
    } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
      if (ret.precision == QUDA_DOUBLE_PRECISION) {
	      packQDPParitySpinor((double2*)packedSpinor1, (double*)spinor);
        printf("double2 packing QDP style\n");
      } else {
      	if (cpu_prec == QUDA_DOUBLE_PRECISION) packQDPParitySpinor((float4*)packedSpinor1, (double*)spinor);
      	else packQDPParitySpinor((float4*)packedSpinor1, (float*)spinor);
      }
    }
    cudaMemcpy(ret.spinor, packedSpinor1, spinor_bytes, cudaMemcpyHostToDevice);
    printf("cudaMemcpy completed\n");
  } else {
    ParitySpinor tmp = allocateParitySpinor(ret.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    loadParitySpinor(tmp, spinor, cpu_prec, dirac_order);
    copyCuda(ret, tmp);
    freeParitySpinor(tmp);
  }

}

void loadFullSpinor(FullSpinor ret, void *spinor, Precision cpu_prec) {

  if (ret.even.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (ret.even.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh_5d*spinorSiteSize*sizeof(double);
    else spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float);
    
#ifndef __DEVICE_EMULATION__
    if (!packedSpinor1) cudaMallocHost(&packedSpinor1, spinor_bytes);
    if (!packedSpinor2) cudaMallocHost(&packedSpinor2, spinor_bytes);
#else
    if (!packedSpinor1) packedSpinor1 = malloc(spinor_bytes);
    if (!packedSpinor2) packedSpinor2 = malloc(spinor_bytes);
#endif
    
    if (ret.even.precision == QUDA_DOUBLE_PRECISION) {
      packFullSpinor((double2*)packedSpinor1, (double2*)packedSpinor2, (double*)spinor);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) 
	packFullSpinor((float4*)packedSpinor1, (float4*)packedSpinor2, (double*)spinor);
      else 
	packFullSpinor((float4*)packedSpinor1, (float4*)packedSpinor2, (float*)spinor);
    }
    
    cudaMemcpy(ret.even.spinor, packedSpinor1, spinor_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ret.odd.spinor, packedSpinor2, spinor_bytes, cudaMemcpyHostToDevice);

#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedSpinor2);
#else
    free(packedSpinor2);
#endif
    packedSpinor2 = 0;
  } else {
    FullSpinor tmp = allocateSpinorField(2*ret.even.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    loadFullSpinor(tmp, spinor, cpu_prec);
    copyCuda(ret.even, tmp.even);
    copyCuda(ret.odd, tmp.odd);
    freeSpinorField(tmp);
  }

}

void loadSpinorField(FullSpinor ret, void *spinor, Precision cpu_prec, DiracFieldOrder dirac_order) {
  void *spinor_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) spinor_odd = (float*)spinor + Nh_5d*spinorSiteSize;
  else spinor_odd = (double*)spinor + Nh_5d*spinorSiteSize;

  if (dirac_order == QUDA_LEX_DIRAC_ORDER) {
    loadFullSpinor(ret, spinor, cpu_prec);
  } else if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
    loadParitySpinor(ret.even, spinor, cpu_prec, dirac_order);
    loadParitySpinor(ret.odd, spinor_odd, cpu_prec, dirac_order);
  } else if (dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    // odd-even so reverse order
    loadParitySpinor(ret.even, spinor_odd, cpu_prec, dirac_order);
    loadParitySpinor(ret.odd, spinor, cpu_prec, dirac_order);
  } else {
    printf("DiracFieldOrder %d not supported\n", dirac_order);
    exit(-1);
  }
}

void retrieveParitySpinor(void *res, ParitySpinor spinor, Precision cpu_prec, DiracFieldOrder dirac_order) {

  if (spinor.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (spinor.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh_5d*spinorSiteSize*sizeof(double);
    else if (spinor.precision == QUDA_SINGLE_PRECISION) spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float);
    else spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float)/2;
    
    if (!packedSpinor1) cudaMallocHost((void**)&packedSpinor1, spinor_bytes);
    cudaMemcpy(packedSpinor1, spinor.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
    if (dirac_order == QUDA_DIRAC_ORDER || QUDA_CPS_WILSON_DIRAC_ORDER) {
      if (spinor.precision == QUDA_DOUBLE_PRECISION) {
	unpackParitySpinor((double*)res, (double2*)packedSpinor1);
      } else {
	if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackParitySpinor((double*)res, (float4*)packedSpinor1);
	else unpackParitySpinor((float*)res, (float4*)packedSpinor1);
      }
    } else if (dirac_order == QUDA_QDP_DIRAC_ORDER) {
      if (spinor.precision == QUDA_DOUBLE_PRECISION) {
	unpackQDPParitySpinor((double*)res, (double2*)packedSpinor1);
      } else {
	if (cpu_prec == QUDA_DOUBLE_PRECISION) unpackQDPParitySpinor((double*)res, (float4*)packedSpinor1);
	else unpackQDPParitySpinor((float*)res, (float4*)packedSpinor1);
      }
    }
  } else {
    ParitySpinor tmp = allocateParitySpinor(spinor.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    copyCuda(tmp, spinor);
    retrieveParitySpinor(res, tmp, cpu_prec, dirac_order);
    freeParitySpinor(tmp);
  }
}

void retrieveFullSpinor(void *res, FullSpinor spinor, Precision cpu_prec) {

  if (spinor.even.precision != QUDA_HALF_PRECISION) {
    size_t spinor_bytes;
    if (spinor.even.precision == QUDA_DOUBLE_PRECISION) spinor_bytes = Nh_5d*spinorSiteSize*sizeof(double);
    else spinor_bytes = Nh_5d*spinorSiteSize*sizeof(float);
    
    if (!packedSpinor1) cudaMallocHost((void**)&packedSpinor1, spinor_bytes);
    if (!packedSpinor2) cudaMallocHost((void**)&packedSpinor2, spinor_bytes);
    
    cudaMemcpy(packedSpinor1, spinor.even.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(packedSpinor2, spinor.odd.spinor, spinor_bytes, cudaMemcpyDeviceToHost);
    if (spinor.even.precision == QUDA_DOUBLE_PRECISION) {
      unpackFullSpinor((double*)res, (double2*)packedSpinor1, (double2*)packedSpinor2);
    } else {
      if (cpu_prec == QUDA_DOUBLE_PRECISION) 
	unpackFullSpinor((double*)res, (float4*)packedSpinor1, (float4*)packedSpinor2);
      else unpackFullSpinor((float*)res, (float4*)packedSpinor1, (float4*)packedSpinor2);
    }
    
#ifndef __DEVICE_EMULATION__
    cudaFreeHost(packedSpinor2);
#else
    free(packedSpinor2);
#endif
    packedSpinor2 = 0;
  } else {
    FullSpinor tmp = allocateSpinorField(2*spinor.even.length/spinorSiteSize, QUDA_SINGLE_PRECISION);
    copyCuda(tmp.even, spinor.even);
    copyCuda(tmp.odd, spinor.odd);
    retrieveFullSpinor(res, tmp, cpu_prec);
    freeSpinorField(tmp);
  }
}

void retrieveSpinorField(void *res, FullSpinor spinor, Precision cpu_prec, DiracFieldOrder dirac_order) {
  void *res_odd;
  if (cpu_prec == QUDA_SINGLE_PRECISION) res_odd = (float*)res + Nh_5d*spinorSiteSize;
  else res_odd = (double*)res + Nh_5d*spinorSiteSize;

  if (dirac_order == QUDA_LEX_DIRAC_ORDER) {
    retrieveFullSpinor(res, spinor, cpu_prec);
  } else if (dirac_order == QUDA_DIRAC_ORDER || dirac_order == QUDA_QDP_DIRAC_ORDER) {
    retrieveParitySpinor(res, spinor.even, cpu_prec, dirac_order);
    retrieveParitySpinor(res_odd, spinor.odd, cpu_prec, dirac_order);
  } else if (dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    retrieveParitySpinor(res, spinor.odd, cpu_prec, dirac_order);
    retrieveParitySpinor(res_odd, spinor.even, cpu_prec, dirac_order);
  } else {
    printf("DiracFieldOrder %d not supported\n", dirac_order);
    exit(-1);
  }
  
}


