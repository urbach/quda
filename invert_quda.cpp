// Ver. 09.11.a

// 11/2/09:  Uncommented the Mat and MatPc code.  Code in this file looks ok, but
//   dependencies need work.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include <invert_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

//#define DBUG

//#include <blas_reference.h>

FullGauge cudaGaugePrecise; // precise gauge field
FullGauge cudaGaugeSloppy; // sloppy gauge field

void printGaugeParam(QudaGaugeParam *param) {

  printf("Gauge Params:\n");
  printf("X = %d\n", param->X);
  printf("Y = %d\n", param->Y);
  printf("Z = %d\n", param->Z);
  printf("T = %d\n", param->T);
  printf("anisotropy = %e\n", param->anisotropy);
  printf("gauge_order = %d\n", param->gauge_order);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("reconstruct = %d\n", param->reconstruct);
  printf("cuda_prec_sloppy = %d\n", param->cuda_prec_sloppy);
  printf("reconstruct_sloppy = %d\n", param->reconstruct_sloppy);
  printf("gauge_fix = %d\n", param->gauge_fix);
  printf("t_boundary = %d\n", param->t_boundary);
  printf("packed_size = %d\n", param->packed_size);
  printf("gaugeGiB = %e\n", param->gaugeGiB);
}

void printInvertParam(QudaInvertParam *param) {
  printf("kappa = %e\n", param->kappa);
  printf("mass_normalization = %d\n", param->mass_normalization);
  printf("inv_type = %d\n", param->inv_type);
  printf("tol = %e\n", param->tol);
  printf("iter = %d\n", param->iter);
  printf("maxiter = %d\n", param->maxiter);
  printf("matpc_type = %d\n", param->matpc_type);
  printf("solution_type = %d\n", param->solution_type);
  printf("preserve_source = %d\n", param->preserve_source);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("dirac_order = %d\n", param->dirac_order);
  printf("spinorGiB = %e\n", param->spinorGiB);
  printf("gflops = %e\n", param->gflops);
  printf("secs = %f\n", param->secs);
}

void initQuda(int dev)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "No devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  for(int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    fprintf(stderr, "found device %d: %s\n", i, deviceProp.name);
  }

  if(dev<0) {
    dev = deviceCount - 1;
    //dev = 0;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (deviceProp.major < 1) {
    fprintf(stderr, "Device %d does not support CUDA.\n", dev);
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);
  cudaSetDevice(dev);

  cudaGaugePrecise.even = NULL;
  cudaGaugePrecise.odd = NULL;

  cudaGaugeSloppy.even = NULL;
  cudaGaugeSloppy.odd = NULL;

}

void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  // gauge_param is a global object, declared in dslash_test_dwf.c.
  gauge_param = param;

  setCudaGaugeParam();
  if (gauge_param->X != L1 || gauge_param->Y != L2 || gauge_param->Z != L3 || gauge_param->T != L4) {
    printf("QUDA error: dimensions do not match: %d=%d, %d=%d, %d=%d, %d=%d\n", 
	   gauge_param->X, L1, gauge_param->Y, L2, gauge_param->Z, L3, gauge_param->T, L4);
    exit(-1);
  }
  gauge_param->packed_size = (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;

  createGaugeField(&cudaGaugePrecise, h_gauge, gauge_param->reconstruct, gauge_param->cuda_prec);
  //J  Here, 1 << 30 = 2^{30} = 1,073,741,824, which I suppose is 1GB but in
  //J  a power of 2 so that it is how a machine really packs 1GB.
  gauge_param->gaugeGiB = 2.0*cudaGaugePrecise.packedGaugeBytes/ (1 << 30);
  if (gauge_param->cuda_prec_sloppy != gauge_param->cuda_prec ||
      gauge_param->reconstruct_sloppy != gauge_param->reconstruct) {
    createGaugeField(&cudaGaugeSloppy, h_gauge, gauge_param->reconstruct_sloppy, gauge_param->cuda_prec_sloppy);
    gauge_param->gaugeGiB += 2.0*cudaGaugeSloppy.packedGaugeBytes/ (1 << 30);
  } else {
    cudaGaugeSloppy = cudaGaugePrecise;
  }

}

void endQuda()
{
  freeSpinorBuffer();
  freeGaugeField(&cudaGaugePrecise);
  freeGaugeField(&cudaGaugeSloppy);
}

void dslash_dwf_Quda(void *h_out, void *h_in, QudaInvertParam *inv_param, 
  int parity, 
  int dagger, double mferm)
{
  ParitySpinor in = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
 
  // We go here right now.  It allocates packedSpinor1 using
  // cudaMallocHost, which is a buffer on CPU for copying to
  // device memory of GPU.  That copy is done with a
  // cudaMemcpy call.  Note that an efficient packing of the
  // input spinor data is performed prior to the transfer
  // onto device memory.
#ifdef DBUG
  // Test that unpacked spinor matches.
  printf("input\n");
  printSpinorElement(h_in,0,inv_param->cuda_prec);
#endif
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);  // spinor_quda.cpp
#ifdef DBUG
  // Test that unpacked spinor matches.
  retrieveParitySpinor(h_out, in, inv_param->cpu_prec, inv_param->dirac_order);
  printf("output\n");
  printSpinorElement(h_out,0,inv_param->cuda_prec);
#endif
  // dslash_dwf_Cuda is declared in dslash_quda.h, 
  // defined in dslash_quda.cu, and is
  // a wrapper for e.g. dslashS_dwf_Cuda().
  // Note that the fermion bare mass is passed, since
  // it now appears in the hopping term.
  dslash_dwf_Cuda(out, cudaGaugePrecise, in, parity, dagger, mferm);
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(out);
  freeParitySpinor(in);
}

// HERE

// ** Dependency to resolve. **
void MatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger)
{
  ParitySpinor in = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  ParitySpinor tmp = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);
  // Defined in dslash_quda.cu.
  MatPC_dwf_Cuda(out, cudaGaugePrecise, in, inv_param->kappa, tmp, inv_param->matpc_type, 
    dagger, inv_param->mferm);
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

// ** Dependency to resolve. **
void MatPCDagMatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  ParitySpinor in = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  ParitySpinor tmp = allocateParitySpinor(Nh_5d, inv_param->cuda_prec);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);  
  // ** Probably not defined yet. **
  MatPCDagMatPC_dwf_Cuda(out, cudaGaugePrecise, in, inv_param->kappa, tmp, inv_param->matpc_type,
    inv_param->mferm);
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

// ** Dependency exists but needs validation with reference code. **
void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger) {
  FullSpinor in = allocateSpinorField(N_5d, inv_param->cuda_prec);
  FullSpinor out = allocateSpinorField(N_5d, inv_param->cuda_prec);

  loadSpinorField(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);

  // ** These need validation with reference code. **
  dslashXpay_dwf_Cuda(out.odd, cudaGaugePrecise, in.even, 1, dagger, in.odd, -inv_param->kappa,
    inv_param->mferm);
  dslashXpay_dwf_Cuda(out.even, cudaGaugePrecise, in.odd, 0, dagger, in.even, -inv_param->kappa,
    inv_param->mferm);

  retrieveSpinorField(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeSpinorField(out);
  freeSpinorField(in);
}


void invertQuda(void *h_x, void *h_b, QudaInvertParam *param)
{
  invert_param = param;

  if (param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported on cpu\n");
    exit(-1);
  }

  int slenh = Nh_5d*spinorSiteSize;
  param->spinorGiB = (double)slenh*(param->cuda_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double): sizeof(float);
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(1<<30);
  else
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(1<<30);

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  double kappa = param->kappa;
  double mferm = param->mferm;

  FullSpinor b, x;
  ParitySpinor in = allocateParitySpinor(Nh_5d, invert_param->cuda_prec); // source vector
  ParitySpinor out = allocateParitySpinor(Nh_5d, invert_param->cuda_prec); // solution vector
  ParitySpinor tmp = allocateParitySpinor(Nh_5d, invert_param->cuda_prec); // temporary used when applying operator

  if (param->solution_type == QUDA_MAT_SOLUTION) {
    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
      b = allocateSpinorField(N_5d, invert_param->cuda_prec);
    } else {
      b.even = out;
      b.odd = tmp;
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) { x.odd = tmp; x.even = out; }
    else { x.even = tmp; x.odd = out; }

    loadSpinorField(b, h_b, param->cpu_prec, param->dirac_order);

    // multiply the source to get the mass normalization
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(2.0*kappa, b.even);
      axCuda(2.0*kappa, b.odd);
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashXpay_dwf_Cuda(in, cudaGaugePrecise, b.odd, 0, 0, b.even, mferm, kappa);
    } else {
      dslashXpay_dwf_Cuda(in, cudaGaugePrecise, b.even, 1, 0, b.odd, mferm, kappa);
    }

  } else if (param->solution_type == QUDA_MATPC_SOLUTION || 
	     param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION){
    loadParitySpinor(in, h_b, param->cpu_prec, param->dirac_order);

    // multiply the source to get the mass normalization
    if (param->mass_normalization == QUDA_MASS_NORMALIZATION)
      if (param->solution_type == QUDA_MATPC_SOLUTION) 
	axCuda(4.0*kappa*kappa, in);
      else
	axCuda(16.0*pow(kappa,4), in);
  }

  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      // Multiply through by Mdag so that we're solving
      // MPCdag MPC x = MPCdag b, equiv. to MPC x = b.
      copyCuda(out, in);
      MatPC_dwf_Cuda(in, cudaGaugePrecise, out, kappa, tmp, param->matpc_type, QUDA_DAG_YES, mferm);
    }
    invertCgCuda(out, in, cudaGaugeSloppy, tmp, param); // inv_cg_quda.cpp
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      invertBiCGstabCuda(out, in, cudaGaugeSloppy, cudaGaugePrecise, tmp, param, QUDA_DAG_YES);
      copyCuda(in, out);
    }
    invertBiCGstabCuda(out, in, cudaGaugeSloppy, cudaGaugePrecise, tmp, param, QUDA_DAG_NO);
    break;
  default:
    printf("Inverter type %d not implemented\n", param->inv_type);
    exit(-1);
  }

  if (param->solution_type == QUDA_MAT_SOLUTION) {

    if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
      // qdp dirac fields are even-odd ordered
      b.even = in;
      loadSpinorField(b, h_b, param->cpu_prec, param->dirac_order);
    }

    if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
      dslashXpay_dwf_Cuda(x.odd, cudaGaugePrecise, out, 1, 0, b.odd, mferm, kappa);
    } else {
      dslashXpay_dwf_Cuda(x.even, cudaGaugePrecise, out, 0, 0, b.even, mferm, kappa);
    }

    retrieveSpinorField(h_x, x, param->cpu_prec, param->dirac_order);

    if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeSpinorField(b);

  } else {
    retrieveParitySpinor(h_x, out, param->cpu_prec, param->dirac_order);
  }

  freeParitySpinor(tmp);
  freeParitySpinor(in);
  freeParitySpinor(out);

  return;
}


