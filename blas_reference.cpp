#include <quda.h>
#include <blas_reference.h>

// performs the operation x[i] *= a
template <typename Float>
void aX(Float a, Float *x, int len) {
  for (int i=0; i<len; i++) x[i] *= a;
}

void ax(double a, void *x, int len, Precision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) aX(a, (double*)x, len);
  else aX((float)a, (float*)x, len);
}

// performs the operation y[i] -= x[i] (minus x plus y)
template <typename Float>
void mXpY(Float *x, Float *y, int len) {
  for (int i=0; i<len; i++) y[i] -= x[i];
}

void mxpy(void* x, void* y, int len, Precision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) mXpY((double*)x, (double*)y, len);
  else mXpY((float*)x, (float*)y, len);
}


// returns the square of the L2 norm of the vector
template <typename Float>
double norm2(Float *v, int len) {
  double sum=0.0;
  for (int i=0; i<len; i++) sum += v[i]*v[i];
  return sum;
}

double norm_2(void *v, int len, Precision precision) {
  if (precision == QUDA_DOUBLE_PRECISION) return norm2((double*)v, len);
  else return norm2((float*)v, len);
}

/*The Fermilab relative residue */
template <typename Float>
double
relative_norm2(Float *p, Float* q,  int len) 
{
    double residue = 0.0;
    for (int i=0; i<len/spinorSiteSize; i++){
	double num=0.0;
	double den=0.0;
	num += p[6*i]*p[6*i];
	num += p[6*i+1]*p[6*i+1];
	num += p[6*i+2]*p[6*i+2];
	num += p[6*i+3]*p[6*i+3];
	num += p[6*i+4]*p[6*i+4];
	num += p[6*i+5]*p[6*i+5];
	
	den += q[6*i]*q[6*i];
	den += q[6*i+1]*q[6*i+1];
	den += q[6*i+2]*q[6*i+2];
	den += q[6*i+3]*q[6*i+3];
	den += q[6*i+4]*q[6*i+4];
	den += q[6*i+5]*q[6*i+5];

	residue += (den ==0)?1.0: (num/den);
    }
    
    int volume = len / spinorSiteSize;
    residue = sqrt(residue/volume);
    return residue;
}
double
relative_norm_2(void* p, void *q, int len, Precision precision) 
{
    if (precision == QUDA_DOUBLE_PRECISION) {
	return relative_norm2((double*)p,(double*)q, len);
    }
    else {
	return relative_norm2((float*)p, (float*)q, len);
    }
}

/*


// sets all elements of the destination vector to zero
void zero(float* a, int cnt) {
    for (int i = 0; i < cnt; i++)
        a[i] = 0;
}

// copy one spinor to the other
void copy(float* a, float *b, int len) {
  for (int i = 0; i < len; i++) a[i] = b[i];
}

// performs the operation y[i] = a*x[i] + b*y[i]
void axpby(float a, float *x, float b, float *y, int len) {
    for (int i=0; i<len; i++) y[i] = a*x[i] + b*y[i];
}

// performs the operation y[i] = a*x[i] + y[i]
void axpy(float a, float *x, float *y, int len) {
    for (int i=0; i<len; i++) y[i] += a*x[i];
}


// returns the real part of the dot product of 2 complex valued vectors
float reDotProduct(float *v1, float *v2, int len) {

  float dot=0.0;
  for (int i=0; i<len; i++) {
    dot += v1[i]*v2[i];
  }

  return dot;
}

// returns the imaginary part of the dot product of 2 complex valued vectors
float imDotProduct(float *v1, float *v2, int len) {

  float dot=0.0;
  for (int i=0; i<len; i+=2) {
    dot += v1[i]*v2[i+1] - v1[i+1]*v2[i];
  }

  return dot;
}

// returns the square of the L2 norm of the vector
double normD(float *v, int len) {

  double sum=0.0;
  for (int i=0; i<len; i++) {
    sum += v[i]*v[i];
  }

  return sum;
}

// returns the real part of the dot product of 2 complex valued vectors
double reDotProductD(float *v1, float *v2, int len) {

  double dot=0.0;
  for (int i=0; i<len; i++) {
    dot += v1[i]*v2[i];
  }

  return dot;
}

// returns the imaginary part of the dot product of 2 complex valued vectors
double imDotProductD(float *v1, float *v2, int len) {

  double dot=0.0;
  for (int i=0; i<len; i+=2) {
    dot += v1[i]*v2[i+1] - v1[i+1]*v2[i];
  }

  return dot;
}
*/
