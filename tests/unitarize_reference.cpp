#include <cstdlib>
#include <cstdio>
#include <math.h> 
#include <string.h>

#include <quda.h> 
#include <test_util.h> 
#include "unitarize_reference.h"
#include "misc.h" 

#include <quda_internal.h>
#include "face_quda.h"


#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

typedef struct {
  float real;
  float imag;
} fcomplex;


typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct { fcomplex e[3][3]; } f3x3_matrix;
typedef struct { fcomplex c[3]; } f3_vector;
typedef struct { dcomplex e[3][3]; } d3x3_matrix;
typedef struct { dcomplex c[3]; } d3_vector;

#define CADD(a,b,c) { (c).real = (a).real + (b).real; \
	(c).imag = (a).imag + (b).imag; }

#define CSUB(a,b,c) { (c).real = (a).real - (b).real; \
	(c).imag = (a).imag - (b).imag; }

#define CMUL(a,b,c) { (c).real = (a).real*(b).real - (a).imag*(b).imag; \
	(c).imag = (a).real*(b).imag + (a).imag*(b).real; }

#define CMULADD(a,b,c) { (c).real += (a).real*(b).real - (a).imag*(b).imag; \
	(c).imag += (a).real*(b).imag + (a).imag*(b).real; }

#define CMULSUB(a,b,c,d,e) { (e).real = (a).real*(b).real - (a).imag*(b).imag - (c).real*(d).real + (c).imag*(d).imag; \
	(e).imag = (a).real*(b).imag + (a).imag*(b).real - (c).real*(d).imag - (c).imag*(d).real; }

#define CMULPLUS(a,b,c,d,e) { (e).real = (a).real*(b).real - (a).imag*(b).imag + (c).real*(d).real - (c).imag*(d).imag; \
	(e).imag = (a).real*(b).imag + (a).imag*(b).real + (c).real*(d).imag + (c).imag*(d).real; }

#define CMULPLUS3(a,b,c,d,e,f,g) { (g).real = (a).real*(b).real - (a).imag*(b).imag + (c).real*(d).real - (c).imag*(d).imag + (e).real*(f).real - (e).imag*(f).imag; \
	(g).imag = (a).real*(b).imag + (a).imag*(b).real + (c).real*(d).imag + (c).imag*(d).real + (e).real*(f).imag + (e).imag*(f).real; }

#define CMULPLUS3KAHAN(a,b,c,d,e,f,g) { float correction = 0.0;                            \
  float next_correction, sum, new_sum;               \
  \
  sum = (a).real*(b).real;                           \
  \
  next_correction =  -(a).imag*(b).imag;             \
  new_sum = sum + next_correction;	           \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;			                   \
  \
  next_correction = (c).real*(d).real - correction;  \
  new_sum = sum + next_correction;	           \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;			                   \
  \
  next_correction = -(c).imag*(d).imag - correction; \
  new_sum = sum + next_correction;	           \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;			                   \
  \
  next_correction = (e).real*(f).real - correction;  \
  new_sum = sum + next_correction;	           \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;			                   \
  \
  next_correction = -(e).imag*(f).imag - correction; \
  new_sum = sum + next_correction;	           \
  correction = (new_sum - sum) - next_correction;    \
  (g).real = new_sum;			           \
  \
  \
  correction = 0.0;		                   \
  sum = (a).real*(b).imag;                           \
  \
  next_correction =   (a).imag*(b).real;             \
  new_sum = sum + next_correction;                   \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;                                     \
  \
  next_correction = (c).real*(d).imag - correction;  \
  new_sum = sum + next_correction;                   \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;                                     \
  \
  next_correction =  (c).imag*(d).real - correction; \
  new_sum = sum + next_correction;                   \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;                                     \
  \
  next_correction = (e).real*(f).imag - correction;  \
  new_sum = sum + next_correction;                   \
  correction = (new_sum - sum) - next_correction;    \
  sum = new_sum;                                     \
  \
  next_correction =  (e).imag*(f).real - correction; \
  new_sum = sum + next_correction;                   \
  correction = (new_sum - sum) - next_correction;    \
  (g).imag = new_sum;                                \
  \
};





#define CDIV(a,b,c) { double t = (b).real*(b).real + (b).imag*(b).imag; \
                      (c).real = ((a).real*(b).real + (a).imag*(b).imag)/t; \
                      (c).imag = ((a).imag*(b).real - (a).real*(b).imag)/t; }

#define CINV(a,b) { double max, ratio; \
		    if(fabs((a).real) > fabs((a).imag)){ max = (a).real; ratio = (a).imag/max; } else{ max = (a).imag; ratio = (a).real/max; } \
		    double t = max*max*(1.0 + ratio*ratio); \
		    (b).real = (a).real/t; \
		    (b).imag = -(a).imag/t; }



#define CSUM(a,b) { (a).real += (b).real; (a).imag += (b).imag; }

#define CMULJ_(a,b,c) { (c).real = (a).real*(b).real + (a).imag*(b).imag; \
	(c).imag = (a).real*(b).imag - (a).imag*(b).real; }


template<typename x3_matrix>
void print_matrix(x3_matrix a){
  int i, j;

  for(int i=0; i<3; ++i){ 
    for(int j=0; j<3; ++j){
	printf("%.10lf, %.10lf\t",a.e[i][j].real,a.e[i][j].imag);
    }
    printf("\n");
  }
  return;
}


template<typename x3_matrix>
void 
unit_matrix(x3_matrix* a){
  for(int i=0; i<3; i++){
    a->e[i][i].real = 1.;
    a->e[i][i].imag = 0.;
    for(int j=0; j<i; ++j){
      a->e[i][j].real = a->e[i][j].imag = a->e[j][i].real = a->e[j][i].imag = 0.;
    }
  }
  return;
}


template<typename x3_matrix, typename cmplx>
void
compute_trace(x3_matrix* m, cmplx* trace){
  trace->real = m->e[0][0].real;
  trace->imag = m->e[0][0].imag;
  for(int i=1; i<3; ++i){ 
    trace->real += m->e[i][i].real;
    trace->imag += m->e[i][i].imag;
  }
}



template<typename x3_matrix, typename cmplx> 
static void compute_det(const x3_matrix* const m, cmplx* det){
  cmplx temp;
  
  CMULSUB(m->e[1][1], m->e[2][2], m->e[1][2], m->e[2][1], temp);
  CMUL(m->e[0][0],temp,*det);

  CMULSUB(m->e[1][2], m->e[2][0], m->e[2][2], m->e[1][0], temp);
  CMULADD(m->e[0][1],temp,*det);

  CMULSUB(m->e[1][0], m->e[2][1], m->e[1][1], m->e[2][0], temp);
  CMULADD(m->e[0][2],temp,*det);

  return;
}



template<typename x3_matrix>
static void compute_inverse(x3_matrix* m, x3_matrix* m_inv){
  
  typeof(m->e[0][0]) temp, det, det_inv;

  CMULSUB(m->e[1][1], m->e[2][2], m->e[1][2], m->e[2][1], m_inv->e[2][2]);
  CMULSUB(m->e[1][2], m->e[2][0], m->e[1][0], m->e[2][2], m_inv->e[0][1]);
  CMULSUB(m->e[1][0], m->e[2][1], m->e[1][1], m->e[2][0], m_inv->e[0][2]);

  // CMULPLUS3KAHAN(m->e[0][0],m_inv->e[2][2],m->e[0][1],m_inv->e[0][1],m->e[0][2],m_inv->e[0][2],det);
  CMULPLUS3(m->e[0][0],m_inv->e[2][2],m->e[0][1],m_inv->e[0][1],m->e[0][2],m_inv->e[0][2],det);
 
  CINV(det, det_inv);

  CMUL(m_inv->e[2][2], det_inv, m_inv->e[0][0]);
  CMUL(m_inv->e[0][1], det_inv, m_inv->e[1][0]);
  CMUL(m_inv->e[0][2], det_inv, m_inv->e[2][0]);

  CMULSUB( m->e[0][2], m->e[2][1], m->e[0][1], m->e[2][2], temp);
  CMUL(temp, det_inv, m_inv->e[0][1]);

  CMULSUB( m->e[0][1], m->e[1][2], m->e[0][2], m->e[1][1], temp);
  CMUL(temp, det_inv, m_inv->e[0][2]);

  CMULSUB( m->e[0][0], m->e[2][2], m->e[0][2], m->e[2][0], temp);
  CMUL(temp, det_inv, m_inv->e[1][1]);

  CMULSUB( m->e[0][2], m->e[1][0], m->e[0][0], m->e[1][2], temp);
  CMUL(temp, det_inv, m_inv->e[1][2]);

  CMULSUB( m->e[0][1], m->e[2][0], m->e[0][0], m->e[2][1], temp);
  CMUL(temp, det_inv, m_inv->e[2][1]);

  CMULSUB( m->e[0][0], m->e[1][1], m->e[0][1], m->e[1][0], temp);
  CMUL(temp, det_inv, m_inv->e[2][2]);

  return;
}






template<typename x3_matrix> 
static void 
unitarize_mult_3x3_nn(const x3_matrix* const a, const x3_matrix* const b, x3_matrix* c)
{
  int i, j, k;
  typeof(a->e[0][0]) x,y;
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      x.real=x.imag=0.0;
      for(k=0;k<3;k++){
	CMUL(a->e[i][k],b->e[k][j],y);
	CSUM(x,y);
      }
      c->e[i][j] = x;
    }
  }
}



template<typename x3_matrix, typename real>
static void 
unitarize_mult_3x3_sn(real scalar, const x3_matrix* const a, x3_matrix* b)
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++){ b->e[i][j].real = scalar*a->e[i][j].real; b->e[i][j].imag = scalar*a->e[i][j].imag; }
}



template<typename x3_matrix> 
static void
unitarize_mult_3x3_an(x3_matrix* a, x3_matrix* b, x3_matrix* c)
{
  int i, j, k;
  typeof(a->e[0][0]) x, y;
  for(i=0;i<3;++i){
    for(j=0;j<3;++j){
      x.real=x.imag=0.0;
      for(k=0;k<3;k++){
	CMULJ_(a->e[k][i],b->e[k][j],y);
  	CSUM(x,y);
      }
      c->e[i][j] = x;
    }
  }
}


template<typename x3_matrix>
static void 
unitarize_sum_3x3_nn(x3_matrix* a, const x3_matrix* const b)
{
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++){ a->e[i][j].real += b->e[i][j].real; a->e[i][j].imag += b->e[i][j].imag; }
}



template<typename x3_matrix> 
static void 
unitarize_adjoint(const x3_matrix* const m, x3_matrix* adj_m){

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
       adj_m->e[i][j].real =  m->e[j][i].real;
       adj_m->e[i][j].imag = -m->e[j][i].imag;	
    }
  }
  return;
}

template<typename x3_matrix>
static void
unitarize_assign(const x3_matrix* const a, x3_matrix* b){

  for(int i=0; i<3; i++){
    for(int j=0; j<3; j++){
      b->e[i][j].real = a->e[i][j].real;
      b->e[i][j].imag = a->e[i][j].imag;
    }
  }
  return;
}



extern int Z[4];
extern int V;
extern int Vh;

template<typename x3_matrix>
static void unitarize_single_link_si(x3_matrix* ulink, const x3_matrix* const fatlink){
  x3_matrix v_inv;
	
  unitarize_assign(fatlink,ulink);
  for(int i=0; i<18; i++){

    compute_inverse(ulink, &v_inv);

    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	  ulink->e[i][j].real += v_inv.e[j][i].real;
	  ulink->e[i][j].real /= 2.;

	  ulink->e[i][j].imag -= v_inv.e[j][i].imag;
	  ulink->e[i][j].imag /= 2.;
      }
    }
  }
  return;
}



#define UNITARIZE_EPS 1e-5
#define PI 3.1415926535897932
#define PI23 2.*PI/3.

/*
template<typename x3_matrix> 
void unitarize_single_link(x3_matrix* ulink, x3_matrix* fatlink){

  x3_matrix q, q2, q3, templink;

  unitarize_mult_3x3_an(fatlink, fatlink, &q);
  unitarize_mult_3x3_nn(&q, &q, &q2);
  unitarize_mult_3x3_nn(&q, &q2, &q3);  

  typeof(fatlink->e[0][0]) tr0, tr1, tr2;
  typeof(fatlink->e[0][0].real) c[3];

  compute_trace(&q,  &tr0);  c[0] = tr0.real;
  compute_trace(&q2, &tr1);  c[1] = tr1.real/2.0;
  compute_trace(&q3, &tr2);  c[2] = tr2.real/3.0;

//   Compute gn 
  typeof(c[0]) s, r, cosTheta, theta, temporary, g[3];
  s = c[1]/3. - c[0]*c[0]/18.;
  g[0] = g[1] = g[2] = c[0]/3.;

  if(fabs(s)>UNITARIZE_EPS) {
     r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);
     cosTheta = r/sqrt(s*s*s);
     if(fabs(cosTheta)>1.0){ r>0 ? theta=0.0 : theta=PI; } 
     else{ theta = acos( cosTheta ); }
//   Compute the eigenvalues of Q 
     temporary = theta/3.;
     for(int i=0;i<3;i++){ g[i] += 2.*sqrt(s)*cos(temporary + (i-1)*PI23); } //printf("g[i] =%lf\n",g[i]); }
  }

  typeof(c[0]) gsqrt[3], u, v, w, f[3], denominator;
  for(int i=0;i<3;i++){ gsqrt[i] = sqrt(g[i]); }
  u = gsqrt[0] + gsqrt[1] + gsqrt[2];
  v = gsqrt[0]*gsqrt[1] + gsqrt[0]*gsqrt[2] + gsqrt[1]*gsqrt[2];
  w = gsqrt[0]*gsqrt[1]*gsqrt[2];

  denominator = w*(u*v-w);
  f[0] = (u*v*v -w*(u*u+v))/denominator;
  f[1] = (-u*u*u -w  + 2.*u*v)/denominator;
  f[2] = u/denominator;
 

//  Now compute 1/sqrt[Q] = f[0]*identity + f[1]*Q + f[2]*Q*Q 
  unit_matrix(&templink);
  unitarize_mult_3x3_sn(f[0],&templink,&templink);
  unitarize_mult_3x3_sn(f[1],&q,&q);
  unitarize_mult_3x3_sn(f[2],&q2,&q2);
  unitarize_sum_3x3_nn(&templink,&q);
  unitarize_sum_3x3_nn(&templink,&q2);

//   Finally, return the unitary link 
  unitarize_mult_3x3_nn(fatlink,&templink,ulink);
  return; 
}
*/
/* The preceding function unitarized a single 3x3 link variable, this 
  routine unitarizes a full gauge field */
template<typename x3_matrix>
static void unitarize_cpu(x3_matrix* ulink, x3_matrix* fatlink)
{

  x3_matrix* f_link = fatlink;
  x3_matrix* u_link = ulink;
  // loop over all lattice sites, even and odd
  for(int i=0; i<V; i++){
    for (int dir=XUP; dir<=TUP; dir++){
      /* fat links and unitarized links are stored contiguously */
      unitarize_single_link_si(u_link++, f_link++); 
    }
  }
  return;
}

#undef PI
#undef PI23
#undef UNITARIZE_EPS



void 
unitarize_reference(void* ulink, void* fatlink, QudaPrecision prec)
{
  switch(prec){ 
    case QUDA_DOUBLE_PRECISION: {
      unitarize_cpu((d3x3_matrix*)ulink, (d3x3_matrix*)fatlink);
      break;
    }
    case QUDA_SINGLE_PRECISION: {
      unitarize_cpu((f3x3_matrix*)ulink, (f3x3_matrix*)fatlink);
      break;
    }
    default:
      fprintf(stderr, "ERROR: unsupported precision\n");
      exit(1);
      break;
  }
  return;
}

