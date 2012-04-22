#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <util_quda.h>

#include <test_util.h>
#include <blas_reference.h>
#include <twisted_mass_dslash_reference.h>

int Z[4];
int V;
int Vh;

void setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;
}

template <typename Float>
void sum(Float *dst, Float *a, Float *b, int cnt) {
  for (int i = 0; i < cnt; i++)
    dst[i] = a[i] + b[i];
}

// performs the operation y[i] = x[i] + a*y[i]
template <typename Float>
void xpay(Float *x, Float a, Float *y, int len) {
    for (int i=0; i<len; i++) y[i] = x[i] + a*y[i];
}



template <typename Float>
Float *gaugeLink(int i, int dir, int oddBit, Float **gaugeEven, Float **gaugeOdd) {
  Float **gaugeField;
  int j;
  
  if (dir % 2 == 0) {
    j = i;
    gaugeField = (oddBit ? gaugeOdd : gaugeEven);
  }
  else {
    switch (dir) {
    case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -1); break;
    case 3: j = neighborIndex(i, oddBit, 0, 0, -1, 0); break;
    case 5: j = neighborIndex(i, oddBit, 0, -1, 0, 0); break;
    case 7: j = neighborIndex(i, oddBit, -1, 0, 0, 0); break;
    default: j = -1; break;
    }
    gaugeField = (oddBit ? gaugeEven : gaugeOdd);
  }
  
  return &gaugeField[dir/2][j*(3*3*2)];
}

template <typename Float>
Float *spinorNeighbor(int i, int dir, int oddBit, Float *spinorField) {
  int j;
  switch (dir) {
  case 0: j = neighborIndex(i, oddBit, 0, 0, 0, +1); break;
  case 1: j = neighborIndex(i, oddBit, 0, 0, 0, -1); break;
  case 2: j = neighborIndex(i, oddBit, 0, 0, +1, 0); break;
  case 3: j = neighborIndex(i, oddBit, 0, 0, -1, 0); break;
  case 4: j = neighborIndex(i, oddBit, 0, +1, 0, 0); break;
  case 5: j = neighborIndex(i, oddBit, 0, -1, 0, 0); break;
  case 6: j = neighborIndex(i, oddBit, +1, 0, 0, 0); break;
  case 7: j = neighborIndex(i, oddBit, -1, 0, 0, 0); break;
  default: j = -1; break;
  }
  
  return &spinorField[j*(4*3*2)];
}

template <typename sFloat, typename gFloat>
void dot(sFloat* res, gFloat* a, sFloat* b) {
  res[0] = res[1] = 0;
  for (int m = 0; m < 3; m++) {
    sFloat a_re = a[2*m+0];
    sFloat a_im = a[2*m+1];
    sFloat b_re = b[2*m+0];
    sFloat b_im = b[2*m+1];
    res[0] += a_re * b_re - a_im * b_im;
    res[1] += a_re * b_im + a_im * b_re;
  }
}

template <typename Float>
void su3Transpose(Float *res, Float *mat) {
  for (int m = 0; m < 3; m++) {
    for (int n = 0; n < 3; n++) {
      res[m*(3*2) + n*(2) + 0] = + mat[n*(3*2) + m*(2) + 0];
      res[m*(3*2) + n*(2) + 1] = - mat[n*(3*2) + m*(2) + 1];
    }
  }
}

template <typename sFloat, typename gFloat>
void su3Mul(sFloat *res, gFloat *mat, sFloat *vec) {
  for (int n = 0; n < 3; n++) dot(&res[n*(2)], &mat[n*(3*2)], vec);
}

template <typename sFloat, typename gFloat>
void su3Tmul(sFloat *res, gFloat *mat, sFloat *vec) {
  gFloat matT[3*3*2];
  su3Transpose(matT, mat);
  su3Mul(res, matT, vec);
}

const double projector[8][4][4][2] = {
  {
    {{1,0}, {0,0}, {0,0}, {0,-1}},
    {{0,0}, {1,0}, {0,-1}, {0,0}},
    {{0,0}, {0,1}, {1,0}, {0,0}},
    {{0,1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {0,1}},
    {{0,0}, {1,0}, {0,1}, {0,0}},
    {{0,0}, {0,-1}, {1,0}, {0,0}},
    {{0,-1}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {1,0}},
    {{0,0}, {1,0}, {-1,0}, {0,0}},
    {{0,0}, {-1,0}, {1,0}, {0,0}},
    {{1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,0}, {-1,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {1,0}, {0,0}},
    {{-1,0}, {0,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,-1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,1}},
    {{0,1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,-1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {0,1}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {0,-1}},
    {{0,-1}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {0,1}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {-1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {-1,0}},
    {{-1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {-1,0}, {0,0}, {1,0}}
  },
  {
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}},
    {{1,0}, {0,0}, {1,0}, {0,0}},
    {{0,0}, {1,0}, {0,0}, {1,0}}
  }
};


// todo pass projector
template <typename Float>
void multiplySpinorByDiracProjector(Float *res, int projIdx, Float *spinorIn) {
  for (int i=0; i<4*3*2; i++) res[i] = 0.0;

  for (int s = 0; s < 4; s++) {
    for (int t = 0; t < 4; t++) {
      Float projRe = projector[projIdx][s][t][0];
      Float projIm = projector[projIdx][s][t][1];
      
      for (int m = 0; m < 3; m++) {
	Float spinorRe = spinorIn[t*(3*2) + m*(2) + 0];
	Float spinorIm = spinorIn[t*(3*2) + m*(2) + 1];
	res[s*(3*2) + m*(2) + 0] += projRe*spinorRe - projIm*spinorIm;
	res[s*(3*2) + m*(2) + 1] += projRe*spinorIm + projIm*spinorRe;
      }
    }
  }
}


//
// dslashReference()
//
// if oddBit is zero: calculate odd parity spinor elements (using even parity spinor)
// if oddBit is one:  calculate even parity spinor elements
//
// if daggerBit is zero: perform ordinary dslash operator
// if daggerBit is one:  perform hermitian conjugate of dslash
//
template <typename Float>
void check_P0(Float *res, Float *in)
{
  for (int i=0; i<4*3*2; i++) res[i] = 0.0;

  
  for (int m = 0; m < 3; m++) {
      res[0*(3*2) + m*(2) + 0] += (in[0*(3*2) + m*(2) + 0]+in[3*(3*2) + m*(2) + 1]);
      res[0*(3*2) + m*(2) + 1] += (in[0*(3*2) + m*(2) + 1]-in[3*(3*2) + m*(2) + 0]);
      
      res[1*(3*2) + m*(2) + 0] += (in[1*(3*2) + m*(2) + 0]+in[2*(3*2) + m*(2) + 1]);
      res[1*(3*2) + m*(2) + 1] += (in[1*(3*2) + m*(2) + 1]-in[2*(3*2) + m*(2) + 0]);
      
      res[2*(3*2) + m*(2) + 0] -= (in[1*(3*2) + m*(2) + 1]-in[2*(3*2) + m*(2) + 0]);
      res[2*(3*2) + m*(2) + 1] += (in[1*(3*2) + m*(2) + 0]+in[2*(3*2) + m*(2) + 1]);
      
      res[3*(3*2) + m*(2) + 0] -= (in[0*(3*2) + m*(2) + 1]-in[3*(3*2) + m*(2) + 0]);
      res[3*(3*2) + m*(2) + 1] += (in[0*(3*2) + m*(2) + 0]+in[3*(3*2) + m*(2) + 1]);      
  }

}

template <typename sFloat, typename gFloat>
void dslashReference(sFloat *res, gFloat **gaugeFull, sFloat *spinorField, int oddBit, int daggerBit) {
  for (int i=0; i<Vh*4*3*2; i++) res[i] = 0.0;
  
  gFloat *gaugeEven[4], *gaugeOdd[4];
  for (int dir = 0; dir < 4; dir++) {  
    gaugeEven[dir] = gaugeFull[dir];
    gaugeOdd[dir]  = gaugeFull[dir]+Vh*gaugeSiteSize;
  }
  
  for (int i = 0; i < Vh; i++) {
    for (int dir = 0; dir < 8; dir++) {
      gFloat *gauge = gaugeLink(i, dir, oddBit, gaugeEven, gaugeOdd);
      sFloat *spinor = spinorNeighbor(i, dir, oddBit, spinorField);
      
      sFloat projectedSpinor[4*3*2], gaugedSpinor[4*3*2];
      int projIdx = 2*(dir/2)+(dir+daggerBit)%2;
      multiplySpinorByDiracProjector(projectedSpinor, projIdx, spinor);
      
      for (int s = 0; s < 4; s++) {
	if (dir % 2 == 0)
	  su3Mul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
	else
	  su3Tmul(&gaugedSpinor[s*(3*2)], gauge, &projectedSpinor[s*(3*2)]);
      }
      
      sum(&res[i*(4*3*2)], &res[i*(4*3*2)], gaugedSpinor, 4*3*2);
//sum(&res[i*(4*3*2)], &res[i*(4*3*2)], spinor, 4*3*2);      
//sum(&res[i*(4*3*2)], &res[i*(4*3*2)], projectedSpinor, 4*3*2);
//check_P0(&res[i*(4*3*2)], spinor);
//if(i < 9) printf("\n%d : %f\n", i, res[i*(4*3*2)]);
    }
  }
}

// applies b*(1 + i*a*gamma_5)
template <typename sFloat>
void twistGamma5(sFloat *out, sFloat *in, const int dagger, const sFloat kappa, const sFloat mu, 
		 const QudaTwistFlavorType flavor, const int V, QudaTwistGamma5Type twist) {

  sFloat a=0.0,b=0.0;
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) { // applying the twist
    a = 2.0 * kappa * mu * flavor; // mu already includes the flavor
    b = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) { // applying the inverse twist
    a = -2.0 * kappa * mu * flavor;
    b = 1.0 / (1.0 + a*a);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  if (dagger) a *= -1.0;

  for(int i = 0; i < V; i++) {
    sFloat tmp[24];
    for(int s = 0; s < 4; s++)
      for(int c = 0; c < 3; c++) {
	sFloat a5 = ((s / 2) ? -1.0 : +1.0) * a;	  
	tmp[s * 6 + c * 2 + 0] = b* (in[i * 24 + s * 6 + c * 2 + 0] - a5*in[i * 24 + s * 6 + c * 2 + 1]);
	tmp[s * 6 + c * 2 + 1] = b* (in[i * 24 + s * 6 + c * 2 + 1] + a5*in[i * 24 + s * 6 + c * 2 + 0]);
      }

    for (int j=0; j<24; j++) out[i*24+j] = tmp[j];
  }
  
}

// this actually applies the preconditioned dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
void dslash(void *res, void **gaugeFull, void *spinorField, double kappa, double mu, 
	    QudaTwistFlavorType flavor, int oddBit, int daggerBit,
	    QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (!daggerBit) {
    if (sPrecision == QUDA_DOUBLE_PRECISION) {
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((double*)res, (double**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((double*)res, (float**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      } 
      twistGamma5((double*)res, (double*)res, daggerBit, kappa, mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    } else {
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((float*)res, (double**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((float*)res, (float**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      }
      twistGamma5((float*)res, (float*)res, daggerBit, (float)kappa, (float)mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    }
  } else {
    if (sPrecision == QUDA_DOUBLE_PRECISION) {
      twistGamma5((double*)spinorField, (double*)spinorField, daggerBit, kappa, mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((double*)res, (double**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((double*)res, (float**)gaugeFull, (double*)spinorField, oddBit, daggerBit);
      }
      twistGamma5((double*)spinorField, (double*)spinorField, daggerBit, kappa, mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT);
    } else {
      twistGamma5((float*)spinorField, (float*)spinorField, daggerBit, (float)kappa, (float)mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      if (gPrecision == QUDA_DOUBLE_PRECISION) {
	dslashReference((float*)res, (double**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      } else {
	dslashReference((float*)res, (float**)gaugeFull, (float*)spinorField, oddBit, daggerBit);
      }
      twistGamma5((float*)spinorField, (float*)spinorField, daggerBit, (float)kappa, (float)mu, 
		  flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT);
    }
  }
}

template <typename sFloat, typename gFloat>
void Mat(sFloat *out, gFloat **gauge, sFloat *in, sFloat kappa, sFloat mu, 
	 QudaTwistFlavorType flavor, int daggerBit) {

  sFloat *inEven = in;
  sFloat *inOdd  = in + Vh*spinorSiteSize;
  sFloat *outEven = out;
  sFloat *outOdd = out + Vh*spinorSiteSize;
  
  sFloat *tmp = (sFloat*)malloc(V*spinorSiteSize*sizeof(sFloat));

  // full dslash operator
  dslashReference(outOdd, gauge, inEven, 1, daggerBit);
  dslashReference(outEven, gauge, inOdd, 0, daggerBit);
  // apply the twist term
  twistGamma5(tmp, in, daggerBit, kappa, mu, flavor, V, QUDA_TWIST_GAMMA5_DIRECT);

  // combine
  xpay(tmp, -kappa, out, V*spinorSiteSize);

  free(tmp);
}

void mat(void *out, void **gauge, void *in, double kappa, double mu, 
	 QudaTwistFlavorType flavor, int dagger_bit,
	 QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (sPrecision == QUDA_DOUBLE_PRECISION)
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      Mat((double*)out, (double**)gauge, (double*)in, (double)kappa, (double)mu, flavor, dagger_bit);
    else 
      Mat((double*)out, (float**)gauge, (double*)in, (double)kappa, (double)mu, flavor, dagger_bit);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      Mat((float*)out, (double**)gauge, (float*)in, (float)kappa, (float)mu, flavor, dagger_bit);
    else 
      Mat((float*)out, (float**)gauge, (float*)in, (float)kappa, (float)mu, flavor, dagger_bit);
}

template <typename Float>
double norm2(Float *v, int len) {
  double sum=0.0;
  for (int i=0; i<len; i++) sum += v[i]*v[i];
  return sum;
}

// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
void MatPC(sFloat *outEven, gFloat **gauge, sFloat *inEven, sFloat kappa, sFloat mu, 
	   QudaTwistFlavorType flavor, int daggerBit, QudaMatPCType matpc_type) {
  
  sFloat *tmp = (sFloat*)malloc(Vh*spinorSiteSize*sizeof(sFloat));
    
  if (!daggerBit) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashReference(tmp, gauge, inEven, 1, daggerBit);
      twistGamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 0, daggerBit);
      twistGamma5(outEven, outEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      dslashReference(tmp, gauge, inEven, 0, daggerBit);
      twistGamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 1, daggerBit);
      twistGamma5(outEven, outEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      twistGamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(tmp, gauge, inEven, 1, daggerBit);
      twistGamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 0, daggerBit);
      twistGamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      twistGamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(tmp, gauge, inEven, 0, daggerBit);
      twistGamma5(tmp, tmp, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven, gauge, tmp, 1, daggerBit);
      twistGamma5(inEven, inEven, daggerBit, kappa, mu, flavor, Vh, QUDA_TWIST_GAMMA5_DIRECT); // undo
    }
  }
  // lastly apply the kappa term
  sFloat kappa2 = -kappa*kappa;
  xpay(inEven, kappa2, outEven, Vh*spinorSiteSize);
  free(tmp);

}

void matpc(void *outEven, void **gauge, void *inEven, double kappa, double mu, QudaTwistFlavorType flavor,
	   QudaMatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (matpc_type != QUDA_MATPC_EVEN_EVEN && matpc_type != QUDA_MATPC_ODD_ODD) {
    printf("Only symmetric preconditioning is implemented in reference\n");
    exit(-1);
  }

  if (sPrecision == QUDA_DOUBLE_PRECISION)
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPC((double*)outEven, (double**)gauge, (double*)inEven, (double)kappa, (double)mu, 
	    flavor, dagger_bit, matpc_type);
    else
      MatPC((double*)outEven, (float**)gauge, (double*)inEven, (double)kappa, (double)mu, 
	    flavor, dagger_bit, matpc_type);
  else
    if (gPrecision == QUDA_DOUBLE_PRECISION) 
      MatPC((float*)outEven, (double**)gauge, (float*)inEven, (float)kappa, (float)mu, 
	    flavor, dagger_bit, matpc_type);
    else
      MatPC((float*)outEven, (float**)gauge, (float*)inEven, (float)kappa, (float)mu,
	    flavor, dagger_bit, matpc_type);
}

//BEGIN NEW

// applies c*(1 + i*a*gamma_5*tau_3 + b*tau_1), tau_1, tau_3 are real!
template <typename sFloat>
void ndegTwistGamma5(sFloat *out1, sFloat *out2, sFloat *in1, sFloat *in2, const int dagger, const sFloat kappa, const sFloat mu, 
		 const sFloat epsilon, const int V, QudaTwistGamma5Type twist) {

  sFloat a=0.0, b=0.0, d=0.0;
  if (twist == QUDA_TWIST_GAMMA5_DIRECT) { // applying the twist
    a = 2.0 * kappa * mu; 
    b = -2.0 * kappa * epsilon;
    d = 1.0;
  } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) { // applying the inverse twist
    a = -2.0 * kappa * mu;
    b = 2.0 * kappa * epsilon;
    d = 1.0 / (1.0 + a*a - b*b);
  } else {
    printf("Twist type %d not defined\n", twist);
    exit(0);
  }

  if (dagger) a *= -1.0;

  for(int i = 0; i < V; i++) {
    sFloat tmp1[24];
    sFloat tmp2[24];    
    for(int s = 0; s < 4; s++)
      for(int c = 0; c < 3; c++) {
	sFloat a5 = ((s / 2) ? -1.0 : +1.0) * a;	  
	tmp1[s * 6 + c * 2 + 0] = d* (in1[i * 24 + s * 6 + c * 2 + 0] - a5*in1[i * 24 + s * 6 + c * 2 + 1] + b*in2[i * 24 + s * 6 + c * 2 + 0]);
	tmp1[s * 6 + c * 2 + 1] = d* (in1[i * 24 + s * 6 + c * 2 + 1] + a5*in1[i * 24 + s * 6 + c * 2 + 0] + b*in2[i * 24 + s * 6 + c * 2 + 1]);
	tmp2[s * 6 + c * 2 + 0] = d* (in2[i * 24 + s * 6 + c * 2 + 0] + a5*in2[i * 24 + s * 6 + c * 2 + 1] + b*in1[i * 24 + s * 6 + c * 2 + 0]);
	tmp2[s * 6 + c * 2 + 1] = d* (in2[i * 24 + s * 6 + c * 2 + 1] - a5*in2[i * 24 + s * 6 + c * 2 + 0] + b*in1[i * 24 + s * 6 + c * 2 + 1]);	
      }
    for (int j=0; j<24; j++) out1[i*24+j] = tmp1[j], out2[i*24+j] = tmp2[j];
  }
  
}

void ndeg_twist_gamma5
(void *out1, void *out2, void *in1, void *in2, const int dagger, const double kappa, const double mu, const double epsilon, const int V, QudaTwistGamma5Type twist)
{
    ndegTwistGamma5((double*)out1, (double*)out2, (double*)in1, (double*)in2, dagger, kappa, mu, epsilon, V, twist);
}

// this actually applies the preconditioned dslash, e.g., D_ee^{-1} D_eo or D_oo^{-1} D_oe
void ndeg_dslash(void *res1, void *res2, void **gaugeFull, void *spinorField1, void *spinorField2, double kappa, double mu, 
	         double epsilon, int oddBit, int daggerBit, QudaPrecision sPrecision, QudaPrecision gPrecision) 
{
  
    void *tmp1 = malloc(24 * Vh * sPrecision);
    void *tmp2 = malloc(24 * Vh * sPrecision);

    if(!daggerBit)
    {
      if (sPrecision == QUDA_DOUBLE_PRECISION) 
      {
	printf("\nI'm here!!!\n");
	  dslashReference((double*)tmp1, (double**)gaugeFull, (double*)spinorField1, oddBit, daggerBit);
	  dslashReference((double*)tmp2, (double**)gaugeFull, (double*)spinorField2, oddBit, daggerBit);	
	  ndegTwistGamma5((double*)res1, (double*)res2, (double*)tmp1, (double*)tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
	  //memcpy(res1, tmp1, 24 * Vh * sPrecision);
//          memset(res2, 0, 24*Vh*sPrecision);
	  //memcpy(res2, tmp2, 24 * Vh * sPrecision);	  
      } 
      else //single precision dslash
      {
	  dslashReference((float*)tmp1, (float**)gaugeFull, (float*)spinorField1, oddBit, daggerBit);
	  dslashReference((float*)tmp2, (float**)gaugeFull, (float*)spinorField2, oddBit, daggerBit);	
	  ndegTwistGamma5((float*)res1, (float*)res2, (float*)tmp1, (float*)tmp2, daggerBit, (float)kappa, (float)mu, (float)epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
	  //memcpy(res1, tmp1, 24 * Vh * sPrecision);
	  //memcpy(res2, tmp2, 24 * Vh * sPrecision);	  
      }
    }
    else
    {
      if (sPrecision == QUDA_DOUBLE_PRECISION) 
      {
	  ndegTwistGamma5((double*)tmp1, (double*)tmp2, (double*)spinorField1, (double*)spinorField2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);	
	  dslashReference((double*)res1, (double**)gaugeFull, (double*)tmp1, oddBit, daggerBit);
	  dslashReference((double*)res2, (double**)gaugeFull, (double*)tmp2, oddBit, daggerBit);	
      } 
      else //single precision dslash
      {
	  ndegTwistGamma5((float*)tmp1, (float*)tmp2, (float*)spinorField1, (float*)spinorField2, daggerBit, (float)kappa, (float)mu, (float)epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);	
	  dslashReference((float*)res1, (float**)gaugeFull, (float*)tmp1, oddBit, daggerBit);
	  dslashReference((float*)res2, (float**)gaugeFull, (float*)tmp2, oddBit, daggerBit);	
      }
    }


    free(tmp1);
    free(tmp2);
}

#include <string.h>
// Apply the even-odd preconditioned Dirac operator
template <typename sFloat, typename gFloat>
void ndegMatPC(sFloat *outEven1, sFloat *outEven2, gFloat **gauge, sFloat *inEven1, sFloat *inEven2, sFloat kappa, sFloat mu, 
	   sFloat epsilon, int daggerBit, QudaMatPCType matpc_type) {
  
  sFloat *tmp1 = (sFloat*)malloc(Vh*spinorSiteSize*sizeof(sFloat));
  sFloat *tmp2 = (sFloat*)malloc(Vh*spinorSiteSize*sizeof(sFloat));  
    
  if (!daggerBit) {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashReference(tmp1, gauge, inEven1, 1, daggerBit);
      dslashReference(tmp2, gauge, inEven2, 1, daggerBit);      
      ndegTwistGamma5(tmp1, tmp2,  tmp1, tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven1, gauge, tmp1, 0, daggerBit);
      dslashReference(outEven2, gauge, tmp2, 0, daggerBit);      
      ndegTwistGamma5(outEven1, outEven2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      dslashReference(tmp1, gauge, inEven1, 0, daggerBit);
      dslashReference(tmp2, gauge, inEven2, 0, daggerBit);      
      ndegTwistGamma5(tmp1, tmp2, tmp1, tmp2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven1, gauge, tmp1, 1, daggerBit);
      dslashReference(outEven2, gauge, tmp2, 1, daggerBit);      
      ndegTwistGamma5(outEven1, outEven2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
    }
  } else {
    if (matpc_type == QUDA_MATPC_EVEN_EVEN) {
      ndegTwistGamma5(tmp1, tmp2, inEven1, inEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven1, gauge, tmp1, 1, daggerBit);
      dslashReference(outEven2, gauge, tmp2, 1, daggerBit);
      ndegTwistGamma5(tmp1, tmp2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven1, gauge, tmp1, 0, daggerBit);
      dslashReference(outEven2, gauge, tmp2, 0, daggerBit);      
    } else if (matpc_type == QUDA_MATPC_ODD_ODD) {
      ndegTwistGamma5(tmp1, tmp2, inEven1, inEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven1, gauge, tmp1, 0, daggerBit);
      dslashReference(outEven2, gauge, tmp2, 0, daggerBit);      
      ndegTwistGamma5(tmp1, tmp2, outEven1, outEven2, daggerBit, kappa, mu, epsilon, Vh, QUDA_TWIST_GAMMA5_INVERSE);
      dslashReference(outEven1, gauge, tmp1, 1, daggerBit);
      dslashReference(outEven2, gauge, tmp2, 1, daggerBit);      
    }
  }
  // lastly apply the kappa term
  sFloat kappa2 = -kappa*kappa;
  xpay(inEven1, kappa2, outEven1, Vh*spinorSiteSize);
  xpay(inEven2, kappa2, outEven2, Vh*spinorSiteSize);  
  
  free(tmp1);
  free(tmp2);

}

void ndeg_matpc(void *outEven1, void *outEven2, void **gauge, void *inEven1, void *inEven2, double kappa, double mu, double epsilon,
	   QudaMatPCType matpc_type, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (matpc_type != QUDA_MATPC_EVEN_EVEN && matpc_type != QUDA_MATPC_ODD_ODD) {
    printf("Only symmetric preconditioning is implemented in reference\n");
    exit(-1);
  }

  if (sPrecision == QUDA_DOUBLE_PRECISION)
  {
      ndegMatPC((double*)outEven1, (double*)outEven2, (double**)gauge, (double*)inEven1, (double*)inEven2, (double)kappa, (double)mu, (double)epsilon, dagger_bit, matpc_type);
  }
  else
  {
      ndegMatPC((float*)outEven1, (float*)outEven2, (float**)gauge, (float*)inEven1, (float*)inEven2, (float)kappa, (float)mu, (float) epsilon, dagger_bit, matpc_type);
  }
}

///

template <typename sFloat, typename gFloat>
void ndegMat(sFloat *out1, sFloat *out2, gFloat **gauge, sFloat *in1, sFloat *in2, sFloat kappa, sFloat mu, 
	 sFloat epsilon, int daggerBit) {

  sFloat *inEven1  = in1;
  sFloat *inOdd1   = in1 + Vh*spinorSiteSize;
  sFloat *outEven1 = out1;
  sFloat *outOdd1  = out1 + Vh*spinorSiteSize;

  sFloat *inEven2  = in2;
  sFloat *inOdd2   = in2 + Vh*spinorSiteSize;
  sFloat *outEven2 = out2;
  sFloat *outOdd2  = out2 + Vh*spinorSiteSize;
 
  sFloat *tmp1 = (sFloat*)malloc(V*spinorSiteSize*sizeof(sFloat));
  sFloat *tmp2 = (sFloat*)malloc(V*spinorSiteSize*sizeof(sFloat));

  // full dslash operator
  dslashReference(outOdd1, gauge, inEven1, 1, daggerBit);
  dslashReference(outOdd2, gauge, inEven2, 1, daggerBit);

  dslashReference(outEven1, gauge, inOdd1, 0, daggerBit);
  dslashReference(outEven2, gauge, inOdd2, 0, daggerBit);      

  // apply the twist term
  ndegTwistGamma5(tmp1, tmp2, in1, in2, daggerBit, kappa, mu, epsilon, V, QUDA_TWIST_GAMMA5_DIRECT);

  // combine
  xpay(tmp1, -kappa, out1, V*spinorSiteSize);
  xpay(tmp2, -kappa, out2, V*spinorSiteSize);

  free(tmp1);
  free(tmp2);
}

void ndeg_mat(void *out1, void* out2, void **gauge, void *in1, void *in2,  double kappa, double mu, 
	 double epsilon, int dagger_bit, QudaPrecision sPrecision, QudaPrecision gPrecision) {

  if (sPrecision == QUDA_DOUBLE_PRECISION)
      ndegMat((double*)out1, (double*)out2, (double**)gauge, (double*)in1, (double*)in2, (double)kappa, (double)mu, (double)epsilon, dagger_bit);
  else
      ndegMat((float*)out1, (float*)out2, (float**)gauge, (float*)in1, (float*)in2, (float)kappa, (float)mu, (float)epsilon, dagger_bit);
}



//END NEW
