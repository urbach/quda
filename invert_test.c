#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <quda.h>
#include <util_quda.h>
#include <dslash_reference.h>

// This gets used in hopping terms.
double mferm= 0.5;
// This gets used in diagonal terms.  So-called DWF height.
#define M0_DWF 1.8
double m0_dwf= M0_DWF;
// DWF kappa...see for instance f_dwf_base.C of CPS, or Andrew P.'s notes for mdwf.
double kappa = 0.5/(5.0-M0_DWF);

int main(int argc, char **argv)
{
  int device = 0;

  void *gauge[4];

  QudaGaugeParam Gauge_param;
  QudaInvertParam inv_param;

  Gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;

  Gauge_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  Gauge_param.reconstruct = QUDA_RECONSTRUCT_12;

  Gauge_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  Gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_12;

  Gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  Gauge_param.X = L1;
  Gauge_param.Y = L2;
  Gauge_param.Z = L3;
  Gauge_param.T = L4;
  Gauge_param.anisotropy = 1.0;

  inv_param.inv_type = QUDA_CG_INVERTER;

  Gauge_param.t_boundary = QUDA_PERIODIC_T;
  Gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param = &Gauge_param;
  

  inv_param.kappa = kappa;
  inv_param.mferm = mferm;
  inv_param.tol = 1e-4;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 1e-2;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  size_t gSize = (Gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(N*gaugeSiteSize*gSize);
  }
  construct_gauge_field(gauge, 1, Gauge_param.cpu_prec);

  void *spinorIn = malloc(N_5d*spinorSiteSize*sSize);
  void *spinorOut = malloc(N_5d*spinorSiteSize*sSize);
  void *spinorCheck = malloc(N_5d*spinorSiteSize*sSize);

  int i0 = 0;
  int s0 = 0;
  int c0 = 0;
  construct_spinor_field(spinorIn, 1, i0, s0, c0, inv_param.cpu_prec);

  double time0 = -((double)clock()); // Start the timer

  initQuda(device);
  loadGaugeQuda((void*)gauge, &Gauge_param);

  invertQuda(spinorOut, spinorIn, &inv_param);  // invert_quda.cpp

  time0 += clock(); // stop the timer
  time0 /= CLOCKS_PER_SEC;

  printf("Cuda Space Required. Spinor:%f + Gauge:%f GiB\n", 
	 inv_param.spinorGiB, Gauge_param.gaugeGiB);
  printf("done: %i iter / %g secs = %g gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, 
    Gauge_param.cpu_prec, inv_param.mferm);
  if  (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
    ax(0.5/inv_param.kappa, spinorCheck, N_5d*spinorSiteSize, inv_param.cpu_prec);

  mxpy(spinorIn, spinorCheck, N_5d*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, N_5d*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, N_5d*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  endQuda();

  return 0;
}

