#include "unitarize_utilities.h"
#include <cstdio>
#include <quda_internal.h>
#include <unitarize_quda.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <read_gauge.h>
#include <gauge_quda.h>
#include <force_common.h>


__device__ __constant__ double UNITARIZE_PI;
__device__ __constant__ double UNITARIZE_PI23;
__device__ __constant__ double UNITARIZE_EPS;

namespace hisq {

  template<class Cmplx> 
  __device__ void reciprocalRoot(LinkVariable<Cmplx>* res, const LinkVariable<Cmplx> & q){

    LinkVariable<Cmplx> qsq, tempq;
    qsq = q*q;
    tempq = qsq*q;

    typename RealTypeId<Cmplx>::Type c[3];
    c[0] = getTrace(q).x;
    c[1] = getTrace(qsq).x/2.0;
    c[2] = getTrace(tempq).x/3.0;

    typename RealTypeId<Cmplx>::Type g[3];
    g[0] = g[1] = g[2] = c[0]/3.;
    typename RealTypeId<Cmplx>::Type r,s,theta;
    s = c[1]/3. - c[0]*c[0]/18;
    r = c[2]/2. - (c[0]/3.)*(c[1] - c[0]*c[0]/9.);

    typename RealTypeId<Cmplx>::Type cosTheta = r/sqrt(s*s*s);
    if(fabs(s) < UNITARIZE_EPS){
      cosTheta = 1.;
      s = 0.0; 
    }
    if(fabs(cosTheta)>1.0){ r>0 ? theta=0.0 : theta=UNITARIZE_PI; }
    else{ theta = acos(cosTheta); }
    theta /= 3.;
    s = 2.0*sqrt(s);
    g[0] += s*cos(theta - UNITARIZE_PI23);
    g[1] += s*cos(theta);
    g[2] += s*cos(theta + UNITARIZE_PI23);
    // At this point we have finished with the c's 
    // use these to store sqrt(g)
    c[0] = sqrt(g[0]); c[1] = sqrt(g[1]); c[2] = sqrt(g[2]);
    // done with the g's, use these to store u, v, w
    g[0] = c[0]+c[1]+c[2];
    g[1] = c[0]*c[1] + c[0]*c[2] + c[1]*c[2];
    g[2] = c[0]*c[1]*c[2];

    const typename RealTypeId<Cmplx>::Type & denominator  = g[2]*(g[0]*g[1]-g[2]); 
    c[0] = (g[0]*g[1]*g[1] - g[2]*(g[0]*g[0]+g[1]))/denominator;
    c[1] = (-g[0]*g[0]*g[0] - g[2] + 2.*g[0]*g[1])/denominator;
    c[2] =  g[0]/denominator;

    tempq = c[1]*q + c[2]*qsq;
    // Add a real scalar
    tempq(0,0).x += c[0];
    tempq(1,1).x += c[0];
    tempq(2,2).x += c[0];
	
    *res = tempq;
    return;
 }
   

  // Unitarize the links using Hamilton-Cayley
  template<class Cmplx>
  __global__ void unitarize_links_hc(Cmplx* fatlink_even, Cmplx* fatlink_odd,
                                     Cmplx* ulink_even,   Cmplx* ulink_odd)
  {
    int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;

    Cmplx* fatlink;
    Cmplx* ulink;

    fatlink = fatlink_even;
    ulink   = ulink_even;
    if(mem_idx >= Vh){
      mem_idx = mem_idx - Vh;
      fatlink = fatlink_odd;
      ulink   = ulink_odd;
    }

    LinkVariable<Cmplx> fat, q, rsqrt_q;
    for(int dir=0; dir<4; ++dir){

      loadLinkVariableFromArray(&fat, fatlink, dir, mem_idx, llfat_ga_stride);
      q = conj(fat)*fat;
      reciprocalRoot<Cmplx>(&rsqrt_q, q);
      q = fat*rsqrt_q;

      writeLinkVariableToArray(ulink, q, dir, mem_idx, llfat_ga_stride);
    }
    return;
  } // end unitarize_links_hc






// simple iterative unitarization routine
  template<class Cmplx>
__global__ void unitarize_links_si(Cmplx* fatlink_even, Cmplx* fatlink_odd,
    Cmplx* ulink_even,  Cmplx* ulink_odd,		
    int max_iters)
{
  int mem_idx = blockIdx.x*blockDim.x + threadIdx.x;
  Cmplx* fatlink;
  Cmplx* ulink;
  fatlink = fatlink_even;
  ulink = ulink_even;
  if(mem_idx >= Vh){
    mem_idx = mem_idx - Vh;
    fatlink = fatlink_odd;
    ulink = ulink_odd;
  }

  LinkVariable<Cmplx>  u, uinv;

  for(int dir=0; dir<4; ++dir){
    loadLinkVariableFromArray(&u, fatlink, dir, mem_idx, llfat_ga_stride);
    for(int i=0; i<max_iters; ++i){
      computeLinkInverse(&uinv, u);
      u = 0.5*(u + conj(uinv));	
    }
    writeLinkVariableToArray(ulink, u, dir, mem_idx, llfat_ga_stride);
  } // end loop over dirs
} // end unitarize_links_si

} // end namespace hisq



// unitarize_init_cuda
// performs a subset of the initializations
// of llfat_init_cuda
  void
unitarize_init_cuda(QudaGaugeParam* param)
{
  static int unitarize_init_cuda_flag = 0;
  if (unitarize_init_cuda_flag){
    return;
  }
  unitarize_init_cuda_flag = 1;

  init_kernel_cuda(param);

  double UNITARIZE_EPS  = 1e-5;
  cudaMemcpyToSymbol("UNITARIZE_EPS", &UNITARIZE_EPS, sizeof(double));

  double  UNITARIZE_PI = 3.1415926535897932;
  cudaMemcpyToSymbol("UNITARIZE_PI", &UNITARIZE_PI, sizeof(double));

  double UNITARIZE_PI23 = 2.*UNITARIZE_PI/3;
  cudaMemcpyToSymbol("UNITARIZE_PI23", &UNITARIZE_PI23, sizeof(double));

  const int Vh = param->X[0]*param->X[1]*param->X[2]*param->X[3]/2;

  // Need to define this so that the other routines can use it
  int site_ga_stride = param->site_ga_pad + Vh;
  cudaMemcpyToSymbol("site_ga_stride", &site_ga_stride, sizeof(int));


  int llfat_ga_stride = param->llfat_ga_pad + Vh;
  cudaMemcpyToSymbol("llfat_ga_stride", &llfat_ga_stride, sizeof(int));

  return;
}





void unitarize_cuda_hc(FullGauge cudaOutLink, FullGauge cudaInLink,
                       const QudaGaugeParam* const param)
{
  const int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM, 1, 1);
  
  
  const QudaPrecision prec = cudaInLink.precision;
  if(prec == QUDA_DOUBLE_PRECISION){
    printf("link unitarization using double precision\n");
  }else if(prec == QUDA_SINGLE_PRECISION){
    printf("link unitarization using single precision\n");
  }

  if(prec == QUDA_DOUBLE_PRECISION){
    hisq::unitarize_links_hc<<<gridDim, blockDim>>>((double2*)cudaInLink.even,  (double2*)cudaInLink.odd,
                                                    (double2*)cudaOutLink.even, (double2*)cudaOutLink.odd);

  }else{ // single precision
    hisq::unitarize_links_hc<<<gridDim, blockDim>>>((float2*)cudaInLink.even,  (float2*)cudaInLink.odd,
                                                    (float2*)cudaOutLink.even, (float2*)cudaOutLink.odd);
  }
  return;
}




void unitarize_cuda_si(FullGauge cudaOutLink, FullGauge cudaInLink,
    const QudaGaugeParam* const param, int num_its)
{
  const int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
  dim3 gridDim(volume/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM, 1, 1);


  const QudaPrecision prec = cudaInLink.precision;
  if(prec == QUDA_DOUBLE_PRECISION){
    printf("link unitarization using double precision\n");
  }else if(prec == QUDA_SINGLE_PRECISION){
    printf("link unitarization using single precision\n");
  }

  if(prec == QUDA_DOUBLE_PRECISION){
    hisq::unitarize_links_si<<<gridDim, blockDim>>>((double2*)cudaInLink.even,  (double2*)cudaInLink.odd,
        (double2*)cudaOutLink.even, (double2*)cudaOutLink.odd,
        num_its);

  }else{ // single precision
    hisq::unitarize_links_si<<<gridDim, blockDim>>>((float2*)cudaInLink.even,  (float2*)cudaInLink.odd,
        (float2*)cudaOutLink.even, (float2*)cudaOutLink.odd,
        num_its);
  }
  return;
}
