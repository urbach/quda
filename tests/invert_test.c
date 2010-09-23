#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <blas_reference.h>
#include <dslash_reference.h>

// in a typical application, quda.h is the only QUDA header required
#include <quda.h>

#define CONF_FILE_PATH "../data/conf.1000"

int main(int argc, char **argv)
{
  // set QUDA parameters

  int device = 0; // CUDA device number

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  gauge_param.X[0] = 24;
  gauge_param.X[1] = 24;
  gauge_param.X[2] = 24;
  gauge_param.X[3] = 48;

  gauge_param.anisotropy = 1.0;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_8;
  gauge_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_8;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  int clover_yes = 0; // 0 for plain Wilson, 1 for clover
  int twist_yes = 1;
  
  if (clover_yes) {
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
  } else if (twist_yes){
    inv_param.dslash_type = QUDA_TWISTED_WILSON_DSLASH;    
  } else {
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
  }
  //inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  inv_param.inv_type = QUDA_CG_INVERTER;  

  //double mass = 0;//-0.94;
  inv_param.kappa = 0.160856;//1.0 / (2.0*(1 + 3/gauge_param.anisotropy + mass));
  inv_param.mu = 0.0085;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 10000;
  inv_param.reliable_delta = 1e-2;

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  gauge_param.ga_pad = 24*24*24;
  inv_param.sp_pad = 24*24*24;
  inv_param.cl_pad = 24*24*24;

  if (clover_yes) {
    inv_param.clover_cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }
  
  inv_param.verbosity = QUDA_SILENT;

  // Everything between here and the call to initQuda() is application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  setDims(gauge_param.X);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  ///construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  //read gauge configuration from the file:
  readILDGconfig(gauge, CONF_FILE_PATH, &gauge_param);

  if (clover_yes) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize);
  void *spinorOut = malloc(V*spinorSiteSize*sSize);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize);

  //for site 0:
  int i0 = 0;
  int s0 = 0;
  int c0 = 0;
  construct_spinor_field(spinorIn, 0, i0, s0, c0, inv_param.cpu_prec);
  
  double time0 = -((double)clock()); // start the timer

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // load the clover term, if desired
  if (clover_yes) loadCloverQuda(NULL, clover_inv, &inv_param);
  
  // perform the inversion
  QudaTwistFlavorType twist_flavor_proj = QUDA_TWIST_MNS;

  invertQuda(spinorOut, spinorIn, &inv_param, twist_flavor_proj);

  time0 += clock(); // stop the timer
  time0 /= CLOCKS_PER_SEC;

  printf("Cuda Space Required:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  if (clover_yes) printf("   Clover: %f GiB\n", inv_param.cloverGiB);
  printf("\nDone: %i iter / %g secs = %g gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param.cpu_prec);
  
  if(inv_param.dslash_type == QUDA_TWISTED_WILSON_DSLASH)
  {
    //apply 2 * i * mu * kappa * gamma_5 to output flavor spinor (in DGR basis):
    if(inv_param.cpu_prec == QUDA_DOUBLE_PRECISION)
    {
      double b = 2 * inv_param.mu * inv_param.kappa * twist_flavor_proj;

      double *out = (double*)spinorOut;
      double *check = (double*)spinorCheck;
    
      for(int i = 0; i < V; i++)
      {
	for(int s = 0; s < 4; s++)
	  for(int c = 0; c < 3; c++)
	  {
	    double a = ((s / 2) ? -1.0 : +1.0) * b;	  
	    check[i * 24 + s * 6 + c * 2 + 0] -= a * out[i * 24 + s * 6 + c * 2 + 1];
	    check[i * 24 + s * 6 + c * 2 + 1] += a * out[i * 24 + s * 6 + c * 2 + 0];
	  }
      }
    }
    else if(inv_param.cpu_prec == QUDA_SINGLE_PRECISION)
    {
      float b = 2 * inv_param.mu * inv_param.kappa * twist_flavor_proj;

      float *out   = (float*)spinorOut;
      float *check = (float*)spinorCheck;
    
      int sign[4] = {1, 1, -1, -1};

      for(int i = 0; i < V; i++)
      {
	for(int s = 0; s < 4; s++)
	  for(int c = 0; c < 3; c++)
	  {
	    float a = sign[s] * b;	  
	    check[i * 24 + s * 6 + c * 2 + 0] -= a * out[i * 24 + s * 6 + c * 2 + 1];
	    check[i * 24 + s * 6 + c * 2 + 1] += a * out[i * 24 + s * 6 + c * 2 + 0];
	  }
      }
    }
  }
  if  (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
    ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);

  mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  // finalize the QUDA library
  endQuda();

  return 0;
}
