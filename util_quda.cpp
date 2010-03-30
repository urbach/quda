#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <quda.h>
#include <util_quda.h>
#include <dslash_reference.h>

#include <complex>

using namespace std;
extern int Z[4];
struct timeval startTime;

void stopwatchStart() {
  gettimeofday(&startTime, NULL);
}

double stopwatchReadSeconds() {
  struct timeval endTime;
  gettimeofday( &endTime, 0);
    
  long ds = endTime.tv_sec - startTime.tv_sec;
  long dus = endTime.tv_usec - startTime.tv_usec;
  return ds + 0.000001*dus;
}

#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) / 2.0)
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1))

template <typename Float>
inline short FloatToShort(Float a) {
  //return (short)(a*MAX_SHORT);
  short rtn = (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
  return rtn;
}

template <typename Float>
inline Float shortToFloat(short a) {
  Float rtn = (float)a/SCALE_FLOAT - SHIFT_FLOAT;
  return rtn;
}

template <typename Float>
void printVector(Float *v) 
{
    printf("(%9f,%9f)  (%9f,%9f)  (%9f,%9f)\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

// X indexes the lattice site
void 
printSpinorElement(void *spinor, int X, Precision precision) 
{
    if (precision == QUDA_DOUBLE_PRECISION){
	printVector((double*)spinor+X*spinorSiteSize);
    }
    else{
	printVector((float*)spinor+X*spinorSiteSize);
    }
}

// X indexes the full lattice
void printGaugeElement(void *gauge, int X, Precision precision) {
  if (getOddBit(X) == 0) {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m=0; m<3; m++) printVector((double*)gauge +(X/2)*gaugeSiteSize + m*3*2);
    else
      for (int m=0; m<3; m++) printVector((float*)gauge +(X/2)*gaugeSiteSize + m*3*2);
      
  } else {
    if (precision == QUDA_DOUBLE_PRECISION)
      for (int m = 0; m < 3; m++) printVector((double*)gauge + (X/2+Vh)*gaugeSiteSize + m*3*2);
    else
      for (int m = 0; m < 3; m++) printVector((float*)gauge + (X/2+Vh)*gaugeSiteSize + m*3*2);
  }
}

// returns 0 or 1 if the full lattice index X is even or odd
int getOddBit(int Y) {
  int x4 = Y/(Z[2]*Z[1]*Z[0]);
  int x3 = (Y/(Z[1]*Z[0])) % Z[2];
  int x2 = (Y/Z[0]) % Z[1];
  int x1 = Y % Z[0];
  return (x4+x3+x2+x1) % 2;
}

// a+=b
template <typename Float>
void complexAddTo(Float *a, Float *b) {
  a[0] += b[0];
  a[1] += b[1];
}

// a = b*c
template <typename Float>
void complexProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = b[0]*c[1] + b[1]*c[0];
}

// a = conj(b)*conj(c)
template <typename Float>
void complexConjugateProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] - b[1]*c[1];
    a[1] = -b[0]*c[1] - b[1]*c[0];
}

// a = conj(b)*c
template <typename Float>
void complexDotProduct(Float *a, Float *b, Float *c) {
    a[0] = b[0]*c[0] + b[1]*c[1];
    a[1] = b[0]*c[1] - b[1]*c[0];
}

// a += b*c
template <typename Float>
void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign) {
  a[0] += sign*(b[0]*c[0] - b[1]*c[1]);
  a[1] += sign*(b[0]*c[1] + b[1]*c[0]);
}

// a += conj(b)*c)
template <typename Float>
void accumulateComplexDotProduct(Float *a, Float *b, Float *c) {
    a[0] += b[0]*c[0] + b[1]*c[1];
    a[1] += b[0]*c[1] - b[1]*c[0];
}

template <typename Float>
void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign) {
  a[0] += sign * (b[0]*c[0] - b[1]*c[1]);
  a[1] -= sign * (b[0]*c[1] + b[1]*c[0]);
}

template <typename Float>
void su3Construct12(Float *mat) {
  Float *w = mat+12;
  w[0] = 0.0;
  w[1] = 0.0;
  w[2] = 0.0;
  w[3] = 0.0;
  w[4] = 0.0;
  w[5] = 0.0;
}

// Stabilized Bunk and Sommer
template <typename Float>
void su3Construct8(Float *mat) {
  mat[0] = atan2(mat[1], mat[0]);
  mat[1] = atan2(mat[13], mat[12]);
  for (int i=8; i<18; i++) mat[i] = 0.0;
}

void su3_construct(void *mat, ReconstructType reconstruct, Precision precision) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    if (precision == QUDA_DOUBLE_PRECISION) su3Construct12((double*)mat);
    else su3Construct12((float*)mat);
  } else {
    if (precision == QUDA_DOUBLE_PRECISION) su3Construct8((double*)mat);
    else su3Construct8((float*)mat);
  }
}

// given first two rows (u,v) of SU(3) matrix mat, reconstruct the third row
// as the cross product of the conjugate vectors: w = u* x v*
// 
// 48 flops
template <typename Float>
void su3Reconstruct12(Float *mat, int dir, int ga_idx, int oddBit) {
  Float *u = &mat[0*(3*2)];
  Float *v = &mat[1*(3*2)];
  Float *w = &mat[2*(3*2)];
  w[0] = 0.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0; w[4] = 0.0; w[5] = 0.0;
  accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
  accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
  accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
  accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
  accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
  accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
  
  double u0 = gauge_param->anisotropy;
  double coff= -u0*u0*24;
  {
      int X1h=gauge_param->X[0]/2;
      int X1 =gauge_param->X[0];
      int X2 =gauge_param->X[1];
      int X3 =gauge_param->X[2];
      int X4 =gauge_param->X[3];
      
      int index = fullLatticeIndex(ga_idx, oddBit);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      
      if (dir == 0) {
           if (i4 % 2 == 1){
               coff *= -1;
           }
       }

       if (dir == 1){
           if ((i1+i4) % 2 == 1){
               coff *= -1;
           }
       }
       if (dir == 2){
           if ( (i4+i1+i2) % 2 == 1){
               coff *= -1;
           }
       }
       if (dir == 3){
           if (ga_idx >= (X4-3)*X1h*X2*X3 ){
               coff *= -1;
           }
       }

       //printf("local ga_idx =%d, index=%d, i4,3,2,1 =%d %d %d %d\n", ga_idx, index, i4, i3, i2,i1);

   }


  w[0]*=coff; w[1]*=coff; w[2]*=coff; w[3]*=coff; w[4]*=coff; w[5]*=coff;
  

}

template <typename Float>
void su3Reconstruct8(Float *mat, int dir, int ga_idx, int oddBit) {
  // First reconstruct first row
  Float row_sum = 0.0;
  row_sum += mat[2]*mat[2];
  row_sum += mat[3]*mat[3];
  row_sum += mat[4]*mat[4];
  row_sum += mat[5]*mat[5];

#if 1
  Float u0= -gauge_param->anisotropy*gauge_param->anisotropy*24;
  {
      int X1h=gauge_param->X[0]/2;
      int X1 =gauge_param->X[0];
      int X2 =gauge_param->X[1];
      int X3 =gauge_param->X[2];
      int X4 =gauge_param->X[3];
      
      int index = fullLatticeIndex(ga_idx, oddBit);
      int i4 = index /(X3*X2*X1);
      int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
      int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
      int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
      
      if (dir == 0) {
	  if (i4 % 2 == 1){
	      u0 *= -1;
	  }
      }
      
      if (dir == 1){
	  if ((i1+i4) % 2 == 1){
               u0 *= -1;
           }
       }
       if (dir == 2){
           if ( (i4+i1+i2) % 2 == 1){
               u0 *= -1;
           }
       }
       if (dir == 3){
           if (ga_idx >= (X4-3)*X1h*X2*X3 ){
               u0 *= -1;
           }
       }
       
       //printf("local ga_idx =%d, index=%d, i4,3,2,1 =%d %d %d %d\n", ga_idx, index, i4, i3, i2,i1);
       
   }
#endif


  Float U00_mag = sqrt( (1.f/(u0*u0) - row_sum)>0? (1.f/(u0*u0)-row_sum):0);
  
  mat[14] = mat[0];
  mat[15] = mat[1];

  mat[0] = U00_mag * cos(mat[14]);
  mat[1] = U00_mag * sin(mat[14]);

  Float column_sum = 0.0;
  for (int i=0; i<2; i++) column_sum += mat[i]*mat[i];
  for (int i=6; i<8; i++) column_sum += mat[i]*mat[i];
  Float U20_mag = sqrt( (1.f/(u0*u0) - column_sum) > 0? (1.f/(u0*u0)-column_sum) : 0);

  mat[12] = U20_mag * cos(mat[15]);
  mat[13] = U20_mag * sin(mat[15]);

  // First column now restored

  // finally reconstruct last elements from SU(2) rotation
  Float r_inv2 = 1.0/(u0*row_sum);

  // U11
  Float A[2];
  complexDotProduct(A, mat+0, mat+6);
  complexConjugateProduct(mat+8, mat+12, mat+4);
  accumulateComplexProduct(mat+8, A, mat+2, u0);
  mat[8] *= -r_inv2;
  mat[9] *= -r_inv2;

  // U12
  complexConjugateProduct(mat+10, mat+12, mat+2);
  accumulateComplexProduct(mat+10, A, mat+4, -u0);
  mat[10] *= r_inv2;
  mat[11] *= r_inv2;

  // U21
  complexDotProduct(A, mat+0, mat+12);
  complexConjugateProduct(mat+14, mat+6, mat+4);
  accumulateComplexProduct(mat+14, A, mat+2, -u0);
  mat[14] *= r_inv2;
  mat[15] *= r_inv2;

  // U12
  complexConjugateProduct(mat+16, mat+6, mat+2);
  accumulateComplexProduct(mat+16, A, mat+4, u0);
  mat[16] *= -r_inv2;
  mat[17] *= -r_inv2;
  
}

void su3_reconstruct(void *mat, int dir, int ga_idx, ReconstructType reconstruct, Precision precision, int oddBit) {
    if (reconstruct == QUDA_RECONSTRUCT_12) {
      if (precision == QUDA_DOUBLE_PRECISION) su3Reconstruct12((double*)mat, dir, ga_idx, oddBit);
      else su3Reconstruct12((float*)mat, dir, ga_idx, oddBit);
  } else {
      if (precision == QUDA_DOUBLE_PRECISION) su3Reconstruct8((double*)mat, dir, ga_idx, oddBit);
      else su3Reconstruct8((float*)mat, dir, ga_idx, oddBit);
  }
}

/*
void su3_construct_8_half(float *mat, short *mat_half) {
  su3Construct8(mat);

  mat_half[0] = floatToShort(mat[0] / M_PI);
  mat_half[1] = floatToShort(mat[1] / M_PI);
  for (int i=2; i<18; i++) {
    mat_half[i] = floatToShort(mat[i]);
  }
}

void su3_reconstruct_8_half(float *mat, short *mat_half, int dir, int ga_idx) {

  for (int i=0; i<18; i++) {
    mat[i] = shortToFloat(mat_half[i]);
  }
  mat[0] *= M_PI;
  mat[1] *= M_PI;

  su3Reconstruct8(mat, dir, ga_idx);
  }*/

template <typename Float>
int compareFloats(Float *a, Float *b, int len, double epsilon) {
  for (int i = 0 ; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon){
	printf("ERROR: %dth float does not match,a=%f, b=%f\n", i, a[i], b[i]);
	return 0;
    }
  }
  return 1;
}

int compare_floats(void *a, void *b, int len, double epsilon, Precision precision) {
    if  (precision == QUDA_DOUBLE_PRECISION) return compareFloats((double*)a, (double*)b, len, epsilon);
  else return compareFloats((float*)a, (float*)b, len, epsilon);
}



// given a "half index" i into either an even or odd half lattice (corresponding
// to oddBit = {0, 1}), returns the corresponding full lattice index.
int fullLatticeIndex(int i, int oddBit) {
  int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
  return 2*i + (boundaryCrossings + oddBit) % 2;
}

template <typename Float>
void applyGaugeFieldScaling(Float **gauge, int Vh) 
{
    
    int X1h=gauge_param->X[0]/2;
    int X1 =gauge_param->X[0];
    int X2 =gauge_param->X[1];
    int X3 =gauge_param->X[2];
    int X4 =gauge_param->X[3];
    
    for(int d =0;d < 4;d++){
	for(int i=0;i < V*gaugeSiteSize;i++){
	    gauge[d][i] /=(-24* gauge_param->anisotropy* gauge_param->anisotropy);
	}
    }
    
    // Apply spatial scaling factor (u0) to spatial links
    for (int d = 0; d < 3; d++) {
	
	//even
	for (int i = 0; i < Vh; i++) {
	    
	    int index = fullLatticeIndex(i, 0);
	    int i4 = index /(X3*X2*X1);
	    int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
	    int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
	    int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
	    int sign=1;
	    
	    if (d == 0) {
		if (i4 % 2 == 1){
		    sign= -1;
		}
	    }
	    
	    if (d == 1){
		if ((i4+i1) % 2 == 1){
		    sign= -1;
		}
	    }
	    if (d == 2){
		if ( (i4+i1+i2) % 2 == 1){
		    sign= -1;
		}
	    }
	    
	    for (int j=0;j < 6; j++){
		gauge[d][i*gaugeSiteSize + 12+ j] *= sign;
	    }
	}

	//odd
	for (int i = 0; i < Vh; i++) {
	    int index = fullLatticeIndex(i, 1);
	    int i4 = index /(X3*X2*X1);
	    int i3 = (index - i4*(X3*X2*X1))/(X2*X1);
	    int i2 = (index - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
	    int i1 = index - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;
	    int sign=1;
	    
	    if (d == 0) {
		if (i4 % 2 == 1){
		    sign= -1;
		}
	    }
	    
	    if (d == 1){
		if ((i4+i1) % 2 == 1){
		    sign= -1;
		}
	    }
	    if (d == 2){
		if ( (i4+i1+i2) % 2 == 1){
		    sign = -1;
		}
	    }
	    
	    for (int j=0;j < 6; j++){
		gauge[d][(Vh+i)*gaugeSiteSize + 12 + j] *= sign;
	    }
	}
	
    }
    
    // Apply boundary conditions to temporal links
    if (gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T) {
	for (int j = 0; j < Vh; j++) {
	    int sign =1;
	    if (j >= (X4-3)*X1h*X2*X3 ){
		sign= -1;
	    }
	    
	    for (int i = 0; i < 6; i++) {
		gauge[3][j*gaugeSiteSize+ 12+ i ] *= sign;
		gauge[3][(Vh+j)*gaugeSiteSize+12 +i] *= sign;
	    }
	}
    }
    
    
}

template <typename Float>
void constructUnitGaugeField(Float **res) {
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gaugeSiteSize;
  }
    
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 0; m < 3; m++) {
	for (int n = 0; n < 3; n++) {
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	}
      }
    }
  }
    
  //applyGaugeFieldScaling(res, Vh);
}

template <typename Float>
void constructUnitGaugeField4(Float *res) 
{
    Float *resOdd, *resEven;
    resEven = res;
    resOdd  = res+Vh*gaugeSiteSize*4;
    
    for (int i = 0; i < Vh*4; i++) {
	for (int m = 0; m < 3; m++) {
	    for (int n = 0; n < 3; n++) {
		resEven[i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
		resEven[i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
		resOdd[i*(3*3*2) + m*(3*2) + n*(2) + 0] = (m==n) ? 1 : 0;
		resOdd[i*(3*3*2) + m*(3*2) + n*(2) + 1] = 0.0;
	    }
	}
    }
    //FIXME: not important at this monent to get link right
    applyGaugeFieldScaling(res, Vh);
}

// normalize the vector a
template <typename Float>
void normalize(complex<Float> *a, int len) {
  double sum = 0.0;
  for (int i=0; i<len; i++) sum += norm(a[i]);
  for (int i=0; i<len; i++) a[i] /= sqrt(sum);
}

// orthogonalize vector b to vector a
template <typename Float>
void orthogonalize(complex<Float> *a, complex<Float> *b, int len) {
  complex<double> dot = 0.0;
  for (int i=0; i<len; i++) dot += conj(a[i])*b[i];
  for (int i=0; i<len; i++) b[i] -= (complex<Float>)dot*a[i];
}

template <typename Float> 
void constructGaugeField(Float **res) {
  Float *resOdd[4], *resEven[4];
  for (int dir = 0; dir < 4; dir++) {  
    resEven[dir] = res[dir];
    resOdd[dir]  = res[dir]+Vh*gaugeSiteSize;
  }
    
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) { // last 2 rows
	for (int n = 0; n < 3; n++) { // 3 columns
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
	  resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;                    
	}
      }
      normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);
      
      normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
      orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
      normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

      {
	Float *w = resEven[dir]+(i*3+0)*3*2;
	Float *u = resEven[dir]+(i*3+1)*3*2;
	Float *v = resEven[dir]+(i*3+2)*3*2;
	
	for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

      {
	Float *w = resOdd[dir]+(i*3+0)*3*2;
	Float *u = resOdd[dir]+(i*3+1)*3*2;
	Float *v = resOdd[dir]+(i*3+2)*3*2;
	
	for (int n = 0; n < 6; n++) w[n] = 0.0;
	accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
	accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
	accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
	accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
	accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
	accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
      }

    }
  }
    
  //FIXME: 
  applyGaugeFieldScaling(res, Vh);
  
}


template <typename Float> 
void constructUnitaryGaugeField(Float **res) 
{
    Float *resOdd[4], *resEven[4];
    for (int dir = 0; dir < 4; dir++) {  
	resEven[dir] = res[dir];
	resOdd[dir]  = res[dir]+Vh*gaugeSiteSize;
    }
    
    for (int dir = 0; dir < 4; dir++) {
	for (int i = 0; i < Vh; i++) {
	    for (int m = 1; m < 3; m++) { // last 2 rows
		for (int n = 0; n < 3; n++) { // 3 columns
		    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
		    resEven[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;
		    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 0] = rand() / (Float)RAND_MAX;
		    resOdd[dir][i*(3*3*2) + m*(3*2) + n*(2) + 1] = rand() / (Float)RAND_MAX;                    
		}
	    }
	    normalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), 3);
	    orthogonalize((complex<Float>*)(resEven[dir] + (i*3+1)*3*2), (complex<Float>*)(resEven[dir] + (i*3+2)*3*2), 3);
	    normalize((complex<Float>*)(resEven[dir] + (i*3 + 2)*3*2), 3);
      
	    normalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), 3);
	    orthogonalize((complex<Float>*)(resOdd[dir] + (i*3+1)*3*2), (complex<Float>*)(resOdd[dir] + (i*3+2)*3*2), 3);
	    normalize((complex<Float>*)(resOdd[dir] + (i*3 + 2)*3*2), 3);

	    {
		Float *w = resEven[dir]+(i*3+0)*3*2;
		Float *u = resEven[dir]+(i*3+1)*3*2;
		Float *v = resEven[dir]+(i*3+2)*3*2;
	
		for (int n = 0; n < 6; n++) w[n] = 0.0;
		accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
		accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
		accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
		accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
		accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
		accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
	    }

	    {
		Float *w = resOdd[dir]+(i*3+0)*3*2;
		Float *u = resOdd[dir]+(i*3+1)*3*2;
		Float *v = resOdd[dir]+(i*3+2)*3*2;
	
		for (int n = 0; n < 6; n++) w[n] = 0.0;
		accumulateConjugateProduct(w+0*(2), u+1*(2), v+2*(2), +1);
		accumulateConjugateProduct(w+0*(2), u+2*(2), v+1*(2), -1);
		accumulateConjugateProduct(w+1*(2), u+2*(2), v+0*(2), +1);
		accumulateConjugateProduct(w+1*(2), u+0*(2), v+2*(2), -1);
		accumulateConjugateProduct(w+2*(2), u+0*(2), v+1*(2), +1);
		accumulateConjugateProduct(w+2*(2), u+1*(2), v+0*(2), -1);
	    }

	}
    }
}


void construct_gauge_field(void **gauge, int type, Precision precision) {
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) constructUnitGaugeField((double**)gauge);
    else constructUnitGaugeField((float**)gauge);
   } else {
    if (precision == QUDA_DOUBLE_PRECISION) constructGaugeField((double**)gauge);
    else constructGaugeField((float**)gauge);
  }
}

void 
construct_fat_long_gauge_field(void **fatlink, void** longlink,  int type, Precision precision) 
{
  if (type == 0) {
      if (precision == QUDA_DOUBLE_PRECISION) {
	  constructUnitGaugeField((double**)fatlink);
	  constructUnitGaugeField((double**)longlink);	  
      }else {
	  constructUnitGaugeField((float**)fatlink);
	  constructUnitGaugeField((float**)longlink);	  
      }
  } else {
      if (precision == QUDA_DOUBLE_PRECISION) {
	  constructGaugeField((double**)fatlink);
	  constructGaugeField((double**)longlink);
      }else {
	  constructGaugeField((float**)fatlink);
	  constructGaugeField((float**)longlink);
      }
  }
}
void 
createSiteLinkCPU(void* link,  Precision precision, int phase) 
{
    void* temp[4];
    
    size_t gSize = (precision == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    for(int i=0;i < 4;i++){
	temp[i] = malloc(V*gaugeSiteSize*gSize);
	if (temp[i] == NULL){
	    fprintf(stderr, "Error: malloc failed for temp in function %s\n", __FUNCTION__);
	    exit(1);
	}
    }
    
    if (precision == QUDA_DOUBLE_PRECISION) {
	constructUnitaryGaugeField((double**)temp);
    }else {
	constructUnitaryGaugeField((float**)temp);
    }
        
    for(int i=0;i < V;i++){
	for(int dir=0;dir < 4;dir++){
	    if (precision == QUDA_DOUBLE_PRECISION){
		double** src= (double**)temp;
		double* dst = (double*)link;
		for(int k=0; k < gaugeSiteSize; k++){
		    dst[ (4*i+dir)*gaugeSiteSize + k ] = src[dir][i*gaugeSiteSize + k];

		}
	    }else{
		float** src= (float**)temp;
		float* dst = (float*)link;
		for(int k=0; k < gaugeSiteSize; k++){
		    dst[ (4*i+dir)*gaugeSiteSize + k ] = src[dir][i*gaugeSiteSize + k];
		    
		}
		
	    }
	}
    }

    if(phase){
	
	for(int i=0;i < V;i++){
	    for(int dir =XUP; dir <= TUP; dir++){
		int idx = i;
		int oddBit =0;
		if (i >= Vh) {
		    idx = i - Vh;
		    oddBit = 1;
		}

		int X1 = Z[0];
		int X2 = Z[1];
		int X3 = Z[2];
		int X4 = Z[3];

		int full_idx = fullLatticeIndex(idx, oddBit);
		int i4 = full_idx /(X3*X2*X1);
		int i3 = (full_idx - i4*(X3*X2*X1))/(X2*X1);
		int i2 = (full_idx - i4*(X3*X2*X1) - i3*(X2*X1))/X1;
		int i1 = full_idx - i4*(X3*X2*X1) - i3*(X2*X1) - i2*X1;	    

		double coeff= 1.0;
		switch(dir){
		case XUP:
		    if ( (i4 & 1) == 1){
			coeff *= -1;
		    }
		    break;

		case YUP:
		    if ( ((i4+i1) & 1) == 1){
			coeff *= -1;
		    }
		    break;

		case ZUP:
		    if ( ((i4+i1+i2) & 1) == 1){
			coeff *= -1;
		    }
		    break;
		
		case TUP:
		    if (i4 == (X4-1) ){
			coeff *= -1;
		    }
		    break;

		default:
		    printf("ERROR: wrong dir(%d)\n", dir);
		    exit(1);
		}
	    
	    
		if (precision == QUDA_DOUBLE_PRECISION){
		    double* mylink = (double*)link;
		    mylink = mylink + (4*i + dir)*gaugeSiteSize;
		
		    mylink[12] *= coeff;
		    mylink[13] *= coeff;
		    mylink[14] *= coeff;
		    mylink[15] *= coeff;
		    mylink[16] *= coeff;
		    mylink[17] *= coeff;
		
		}else{
		    float* mylink = (float*)link;
		    mylink = mylink + (4*i + dir)*gaugeSiteSize;
		
		    mylink[12] *= coeff;
		    mylink[13] *= coeff;
		    mylink[14] *= coeff;
		    mylink[15] *= coeff;
		    mylink[16] *= coeff;
		    mylink[17] *= coeff;
		
		}
	    }
	}
    }    

    
#if 1
    for(int i=0;i< 4*V*gaugeSiteSize;i++){
	if (precision ==QUDA_SINGLE_PRECISION){
	    float* f = (float*)link;
	    if (f[i] != f[i] || (fabsf(f[i]) > 1.e+3) ){
		fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
		exit(1);
	    }
	}else{
	    double* f = (double*)link;
	    if (f[i] != f[i] || (fabs(f[i]) > 1.e+3)){
		fprintf(stderr, "ERROR:  %dth: bad number(%f) in function %s \n",i, f[i], __FUNCTION__);
		exit(1);
	    }
	    
	}
	
    }
#endif

    for(int i=0;i < 4;i++){
	free(temp[i]);
    }
    return;
}


void 
createMomCPU(void* mom,  Precision precision) 
{
    void* temp;
    
    size_t gSize = (precision == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    temp = malloc(4*V*gaugeSiteSize*gSize);
    if (temp == NULL){
	fprintf(stderr, "Error: malloc failed for temp in function %s\n", __FUNCTION__);
	exit(1);
    }
    
    
    
    for(int i=0;i < V;i++){
	if (precision == QUDA_DOUBLE_PRECISION){
	    for(int dir=0;dir < 4;dir++){
		double* thismom = (double*)mom;	    
		for(int k=0; k < momSiteSize; k++){
		    thismom[ (4*i+dir)*momSiteSize + k ]= 1.0* rand() /RAND_MAX;				
		}	    
	    }	    
	}else{
	    for(int dir=0;dir < 4;dir++){
		float* thismom=(float*)mom;
		for(int k=0; k < momSiteSize; k++){
		    thismom[ (4*i+dir)*momSiteSize + k ]= 1.0* rand() /RAND_MAX;		
		}	    
	    }
	}
    }
    
    free(temp);
    return;
}

void
createHwCPU(void* hw,  Precision precision)
{
    for(int i=0;i < V;i++){
        if (precision == QUDA_DOUBLE_PRECISION){
            for(int dir=0;dir < 4;dir++){
                double* thishw = (double*)hw;
                for(int k=0; k < hwSiteSize; k++){
                    thishw[ (4*i+dir)*hwSiteSize + k ]= 1.0* rand() /RAND_MAX;
                }
            }
        }else{
            for(int dir=0;dir < 4;dir++){
                float* thishw=(float*)hw;
                for(int k=0; k < hwSiteSize; k++){
                    thishw[ (4*i+dir)*hwSiteSize + k ]= 1.0* rand() /RAND_MAX;
                }
            }
        }
    }

    return;
}


template <typename Float>
void constructPointSpinorField(Float *res, int i0, int s0, int c0) {
  Float *resEven = res;
  Float *resOdd = res + Vh*spinorSiteSize;
    
  for(int i = 0; i < Vh; i++) {
    for (int s = 0; s < 1; s++) {
      for (int m = 0; m < 3; m++) {
	resEven[i*(1*3*2) + s*(3*2) + m*(2) + 0] = 0;
	resEven[i*(1*3*2) + s*(3*2) + m*(2) + 1] = 0;
	resOdd[i*(1*3*2) + s*(3*2) + m*(2) + 0] = 0;
	resOdd[i*(1*3*2) + s*(3*2) + m*(2) + 1] = 0;
	if (s == s0 && m == c0) {
	  if (fullLatticeIndex(i, 0) == i0)
	    resEven[i*(1*3*2) + s*(3*2) + m*(2) + 0] = 1;
	  if (fullLatticeIndex(i, 1) == i0)
	    resOdd[i*(1*3*2) + s*(3*2) + m*(2) + 0] = 1;
	}
      }
    }
  }
}

template <typename Float>
void constructSpinorField(Float *res) {
  for(int i = 0; i < V; i++) {
    for (int s = 0; s < 1; s++) {
      for (int m = 0; m < 3; m++) {
	res[i*(1*3*2) + s*(3*2) + m*(2) + 0] = rand() / (Float)RAND_MAX;
	res[i*(1*3*2) + s*(3*2) + m*(2) + 1] = rand() / (Float)RAND_MAX;
      }
    }
  }
}

void construct_spinor_field(void *spinor, int type, int i0, int s0, int c0, Precision precision) {
  if (type == 0) {
    if (precision == QUDA_DOUBLE_PRECISION) constructPointSpinorField((double*)spinor, i0, s0, c0);
    else constructPointSpinorField((float*)spinor, i0, s0, c0);
  } else {
      if (precision == QUDA_DOUBLE_PRECISION) constructSpinorField((double*)spinor);
    else constructSpinorField((float*)spinor);
  }
}

template <typename Float>
void compareSpinor(Float *spinorRef, Float *spinorGPU, int len) {
  int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[6];
  for (int i=0; i<6; i++) iter[i] = 0;

  for (int i=0; i<len; i++) {
    for (int j=0; j<6; j++) {
      int is = i*6+j;
      double diff = fabs(spinorRef[is]-spinorGPU[is]);
      for (int f=0; f<fail_check; f++)
	if (diff > pow(10.0,-(f+1))) fail[f]++;
      //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
      if (diff > 1e-3) iter[j]++;
    }
  }
    
  for (int i=0; i<6; i++) printf("%d fails = %d\n", i, iter[i]);
    
  for (int f=0; f<fail_check; f++) {
    printf("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*spinorSiteSize, fail[f] / (double)(len*6));
  }

}

void compare_spinor(void *spinor_ref, void *spinor_gpu, int len, Precision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) compareSpinor((double*)spinor_ref, (double*)spinor_gpu, len);
  else compareSpinor((float*)spinor_ref, (float*)spinor_gpu, len);
}

void strong_check(void *spinorRef, void *spinorGPU, int len, Precision prec) {
  printf("Reference:\n");
  printSpinorElement(spinorRef, 0, prec); 
  printf("...\n");
  printSpinorElement(spinorRef, len-1, prec); 
  printf("\n");    
  
  printf("\nCUDA:\n");
  printSpinorElement(spinorGPU, 0, prec); 
  printf("...\n");
  printSpinorElement(spinorGPU, len-1, prec); 
  printf("\n");

  compare_spinor(spinorRef, spinorGPU, len, prec);
}




template <typename Float>
void compareLink(Float *linkA, Float *linkB, int len) {
  int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[18];
  for (int i=0; i<18; i++) iter[i] = 0;
  
  for (int i=0; i<len; i++) {
      for (int j=0; j<18; j++) {
	  int is = i*18+j;
	  double diff = fabs(linkA[is]-linkB[is]);
	  for (int f=0; f<fail_check; f++)
	      if (diff > pow(10.0,-(f+1))) fail[f]++;
	  //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	  if (diff > 1e-3) iter[j]++;
      }
  }
  
  for (int i=0; i<18; i++) printf("%d fails = %d\n", i, iter[i]);
  
  for (int f=0; f<fail_check; f++) {
      printf("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*gaugeSiteSize, fail[f] / (double)(len*6));
  }
  
}

static void 
compare_link(void *linkA, void *linkB, int len, Precision precision)
{
    if (precision == QUDA_DOUBLE_PRECISION) compareLink((double*)linkA, (double*)linkB, len);
    else compareLink((float*)linkA, (float*)linkB, len);
    
    return;
}


// X indexes the lattice site
static void 
printLinkElement(void *link, int X, Precision precision) 
{
    if (precision == QUDA_DOUBLE_PRECISION){
	for(int i=0; i < 3;i++){
	    printVector((double*)link+ X*gaugeSiteSize + i*6);
	}
	
    }
    else{
	for(int i=0;i < 3;i++){
	    printVector((float*)link+X*gaugeSiteSize + i*6);
	}
    }
}

void strong_check_link(void * linkA, void *linkB, int len, Precision prec) 
{
    printf("LinkA:\n");
    printLinkElement(linkA, 0, prec); 
    printf("...\n");
    printLinkElement(linkA, len-1, prec); 
    printf("\n");    
    
    printf("\nlinkB:\n");
    printLinkElement(linkB, 0, prec); 
    printf("...\n");
    printLinkElement(linkB, len-1, prec); 
    printf("\n");
    
    compare_link(linkA, linkB, len, prec);
}

template <typename Float>
void compare_mom(Float *momA, Float *momB, int len) {
  int fail_check = 16;
  int fail[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int iter[momSiteSize];
  for (int i=0; i<momSiteSize; i++) iter[i] = 0;
  
  for (int i=0; i<len; i++) {
      for (int j=0; j<momSiteSize; j++) {
	  int is = i*momSiteSize+j;
	  double diff = fabs(momA[is]-momB[is]);
	  for (int f=0; f<fail_check; f++)
	      if (diff > pow(10.0,-(f+1))) fail[f]++;
	  //if (diff > 1e-1) printf("%d %d %e\n", i, j, diff);
	  if (diff > 1e-3) iter[j]++;
      }
  }
  
  for (int i=0; i<momSiteSize; i++) printf("%d fails = %d\n", i, iter[i]);
  
  for (int f=0; f<fail_check; f++) {
      printf("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)), fail[f], len*momSiteSize, fail[f] / (double)(len*6));
  }
  
}

static void 
printMomElement(void *mom, int X, Precision precision) 
{
    if (precision == QUDA_DOUBLE_PRECISION){
	double* thismom = ((double*)mom)+ X*momSiteSize;
	printVector(thismom);
	printf("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);
    }else{
	float* thismom = ((float*)mom)+ X*momSiteSize;
	printVector(thismom);
	printf("(%9f,%9f) (%9f,%9f)\n", thismom[6], thismom[7], thismom[8], thismom[9]);	
    }
}
void strong_check_mom(void * momA, void *momB, int len, Precision prec) 
{    
    printf("mom:\n");
    printMomElement(momA, 0, prec); 
    printf("\n");
    printMomElement(momA, 1, prec); 
    printf("\n");
    printMomElement(momA, 2, prec); 
    printf("\n");
    printMomElement(momA, 3, prec); 
    printf("...\n");

    printf("\nreference mom:\n");
    printMomElement(momB, 0, prec); 
    printf("\n");
    printMomElement(momB, 1, prec); 
    printf("\n");
    printMomElement(momB, 2, prec); 
    printf("\n");
    printMomElement(momB, 3, prec); 
    printf("\n");

    
    if (prec == QUDA_DOUBLE_PRECISION){
	compare_mom((double*)momA, (double*)momB, len);
    }else{
	compare_mom((float*)momA, (float*)momB, len);
    }
}
