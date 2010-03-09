// dslash_test_dwf.c
// Ver. 09.12.a

// 10/19/09 -- need to step through dslashRef() [in this file] 
//   and dslashCUDA() [in this file] to see where
//   they differ.
// 11/3/09:  turned on Mat and MatPC.  kappa  is
//   passed in these functions
//   since kappa2= -kappa^2 is computed in MatPC_dwf_Cuda()
//   of dslash_quda.cu.

#include <stdio.h>
#include <stdlib.h>

#include <quda.h>  // includes enum_quda.h, invert_quda.h,
  // blas_quda.h, dslash_quda.h
#include <dslash_reference.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat, 3 = MatPCDagMatPC, 4 = MatDagMat)
int test_type = 2;
// Here is a fun project:  compare the speed of MatPC vs. Mat.
// Also, compare the accuracies.  Make sure the solns. are the same.

QudaGaugeParam gaugeParam;  // invert_quda.h
// Used even when we don't do an inversion.
QudaInvertParam inv_param;  // invert_quda.h

FullGauge gauge;  // quda.h
FullSpinor cudaSpinor;  // quda.h
FullSpinor cudaSpinorOut;  // quda.h
ParitySpinor tmp;  // quda.h

void *hostGauge[4];
void *spinor, *spinorRef, *spinorGPU;
void *spinorEven, *spinorOdd;
    
int ODD_BIT = 0;

int DAGGER_BIT = 1;
int TRANSFER = 0; // include transfer time in the benchmark?
// This gets used in hopping terms.
double mferm= 0.1;
// This gets used in diagonal terms.  So-called DWF height.
#define M0_DWF 1.8
double m0_dwf= M0_DWF;
// DWF kappa...see for instance f_dwf_base.C of CPS, or Andrew P.'s notes for mdwf.
double kappa = 0.5/(5.0-M0_DWF);

void init() {

  // -- Here we initialize the global QudaGaugeParam object. --
  // The precision types are defined in enum_quda.h.
  gaugeParam.cpu_prec = QUDA_SINGLE_PRECISION;
  gaugeParam.cuda_prec = QUDA_SINGLE_PRECISION;

  gaugeParam.reconstruct = QUDA_RECONSTRUCT_12;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;
  // Read off parameters that are in quda.h
  gaugeParam.X = L1;
  gaugeParam.Y = L2;
  gaugeParam.Z = L3;
  gaugeParam.T = L4;
  // Gauge fields are 4d, so I haven't added an Ls here.
  //
  // I changed anisotropy from 2.3 to 1 since we want to
  // start with simplest case.
  gaugeParam.anisotropy = 1.0;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  // Q.  Why put this into the gaugeParam?  Isn't is the fermions
  // that are antiperiodic in time?
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  // Q.  Where in the world is gauge_param declared?  Why do we need
  // a pointer to this global object?
  gauge_param = &gaugeParam;

  // Need these because they're used even when we don't do an inversion.
  inv_param.cpu_prec = QUDA_SINGLE_PRECISION;
  inv_param.cuda_prec = QUDA_SINGLE_PRECISION;
  if (test_type == 2) inv_param.dirac_order = QUDA_DIRAC_ORDER;
  else inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.kappa = kappa;
  inv_param.mferm = mferm;
  invert_param = &inv_param;
  

  

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // construct input fields
  // Here we are allocating in CPU ("host") mem resource
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(N*gaugeSiteSize*gSize);
  // Here we are allocating in CPU ("host") mem resource
  spinor = malloc(N_5d*spinorSiteSize*sSize);
  spinorRef = malloc(N_5d*spinorSiteSize*sSize);
  spinorGPU = malloc(N_5d*spinorSiteSize*sSize);
  //J  spinorEven points to beginning of spinor array.
  spinorEven = spinor;
  //J  spinorOdd point to 2nd half of spinor array.
  if (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION)
    spinorOdd = (void*)((double*)spinor + Nh_5d*spinorSiteSize);
  else 
    spinorOdd = (void*)((float*)spinor + Nh_5d*spinorSiteSize);
    
  printf("Randomizing fields...");
  construct_gauge_field(hostGauge, 1, gaugeParam.cpu_prec);
  // Initialize all elements of input spinor to random complex numbers (uniform deviate [0,1]).
  construct_spinor_field(spinor, 1, 0, 0, 0, inv_param.cpu_prec);  // util_quda.cpp
  printf("done.\n"); fflush(stdout);
  
  //J  Now we will presumably start loading the GPGPU.
  //J  XXX HERE XXX
  int dev = 0;
  initQuda(dev);  // invert_quda.h/cpp
  //J  Note that invert_quda.cpp declares global variables
  //J  cudaGaugePrecise and cudaGaugeSloppy.  These come back
  //J  null from initQuda().
  loadGaugeQuda(hostGauge, &gaugeParam);  // invert_quda.h/cpp

  gauge = cudaGaugePrecise;

  printf("Sending fields to GPU..."); fflush(stdout);
// presently we're skipping this block...
  if (!TRANSFER) {

    tmp = allocateParitySpinor(Nh_5d, inv_param.cuda_prec);
    cudaSpinor = allocateSpinorField(N_5d, inv_param.cuda_prec);
    cudaSpinorOut = allocateSpinorField(N_5d, inv_param.cuda_prec);

    if (test_type < 2) {
      loadParitySpinor(cudaSpinor.even, spinorEven, inv_param.cpu_prec, 
		       inv_param.dirac_order); // spinor_quda.cpp
    } else {
      loadSpinorField(cudaSpinor, spinor, inv_param.cpu_prec, 
		      inv_param.dirac_order);
    }
  }
// ...and exiting init().

}

void end() {
  // release memory
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  free(spinorGPU);
  free(spinor);
  free(spinorRef);
  if (!TRANSFER) {
    freeSpinorField(cudaSpinorOut);
    freeSpinorField(cudaSpinor);
    freeParitySpinor(tmp);
  }
  endQuda();
}

// The GPU call.
// -- HERE: Follow this one through. --
double dslashCUDA() {

  // execute kernel
  const int LOOPS = 10;
  printf("Executing %d kernel loops...", LOOPS);
  fflush(stdout);
  stopwatchStart();
  // Q. How do the loops affect output?  The CPU code does not
  // do loops, so why is it expected to come out the same?
  // A. The same input half spinor is used each time.  It is
  // identical each time.
  for (int i = 0; i < LOOPS; i++) {
    switch (test_type) {
    case 0:
      if (TRANSFER) {
        // ** Presently going here. **
        // dslash_dwf_Quda defined in invert_quda.cpp.
        // It does a couple things, then calls dslash_dwf_Cuda().
        // See coments below re. that fcn.
        // spinorEven is input, which is not changed in the
        // call.  spinorOdd is output.  So the loop just
        // does an identical operation 10 times.
        dslash_dwf_Quda(spinorOdd, spinorEven, &inv_param, 
          ODD_BIT, DAGGER_BIT, mferm);
      } else {
        // Only off-diagonal pieces get computed in dslash.
        // For full DWF matrix, must include 5d kappa -- diagonal term.
        // dslash_dwf_Cuda is declared in dslash_quda.h, 
        // defined in dslash_quda.cu, and is
        // a wrapper for e.g. dslashS_dwf_Cuda().
        // Note that the fermion bare mass is passed, since
        // it now appears in the hopping term.
        dslash_dwf_Cuda(cudaSpinor.odd, gauge, 
          cudaSpinor.even, ODD_BIT, DAGGER_BIT,mferm);
      }
      break;
    case 1:
      if (TRANSFER) MatPCQuda(spinorOdd, spinorEven, &inv_param, DAGGER_BIT);
      else MatPC_dwf_Cuda(cudaSpinor.odd, gauge, cudaSpinor.even, kappa, tmp, 
        QUDA_MATPC_EVEN_EVEN, DAGGER_BIT, mferm);
      break;
    case 2:
      if (TRANSFER) MatQuda(spinorGPU, spinor, &inv_param, DAGGER_BIT);
      else Mat_dwf_Cuda(cudaSpinorOut, gauge, cudaSpinor, kappa, DAGGER_BIT, mferm);
    }
  }
    
  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  cudaThreadSynchronize();
  double secs = stopwatchReadSeconds() / LOOPS;
  printf("done.\n\n");

  return secs;
}


void dslashRef() {
  
  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
    case 0:
    
      // dslash_reference.cpp/.h
      dslash_reference_5d(spinorRef, hostGauge, spinorEven, 
         ODD_BIT, DAGGER_BIT, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec, mferm);
    
    break;
  case 1:    
    matpc(spinorRef, hostGauge, spinorEven, kappa,
      QUDA_MATPC_EVEN_EVEN, DAGGER_BIT, 
	  inv_param.cpu_prec, gaugeParam.cpu_prec, mferm);
    break;
  case 2:
    mat(spinorRef, hostGauge, spinor, kappa, DAGGER_BIT, 
	inv_param.cpu_prec, gaugeParam.cpu_prec,mferm);
    break;
  default:
    printf("Test type not defined\n");
    exit(-1);
  }

  printf("done.\n");
    
}

void dslashTest() {

  init(); // followed this through.

  // Here, 1 << 30 is 2^{30}, which is "1 GB", in 2^n.
  // Measure GB of spinor.
  float spinorGiB = (float)Nh_5d*spinorSiteSize*sizeof(inv_param.cpu_prec) 
          / (1 << 30);
  // Similarly, 2^{10} = 1024 = 1 KB.
  // Q. Where is this function defined.
  float sharedKB = (float)dslashCudaSharedBytes() / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  // Q. Where did this get set?  A. In loadGaugeQuda() [invert_quda.cpp],
  // called in init() above.
  printf("Gauge mem: %.3f GiB\n", gaugeParam.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);

  int attempts = 10000;
  // CPU evaluation.  ** TODO:  This one is coming back with zeros in the
  // second array.  Why? **
  dslashRef();
  for (int i=0; i<attempts; i++) {
    
    // ---------------------------------------------------------
    // -- Here is the main call.  Execution time is returned. --
    // ---------------------------------------------------------
    // GPU evaluation.
    double secs = dslashCUDA();
  
    // Presently this block is skipped.
    if (!TRANSFER) {
      // secs is not passed to these, so how is transfer time included in
      // timing?
      if (test_type < 2) retrieveParitySpinor(spinorOdd, 
        cudaSpinor.odd, inv_param.cpu_prec, inv_param.dirac_order);
      else retrieveSpinorField(spinorGPU, cudaSpinorOut, 
        inv_param.cpu_prec, inv_param.dirac_order);
    }
    // print timing information
    printf("%fms per loop\n", 1000*secs);
    int flops = test_type ? 1320*2 + 48 : 1320;
    int floats = test_type ? 2*(7*24+8*gaugeParam.packed_size+24)+24 
      : 7*24+8*gaugeParam.packed_size+24;
    printf("GFLOPS = %f\n", 1.0e-9*flops*Nh_5d/secs);
    printf("GiB/s = %f\n\n", Nh_5d*floats*sizeof(float)/(secs*(1<<30)));
    
    int res;
    // compare_floats is in util_quda.cpp/.h
    if (test_type < 2) res = compare_floats(spinorOdd, spinorRef, Nh_5d*4*3*2,
      1e-4, inv_param.cpu_prec);
    else res = compare_floats(spinorGPU, spinorRef, N_5d*4*3*2, 1e-4, inv_param.cpu_prec);
    // Presently I'm getting a fail here.
    printf("%d Test %s\n", i, (1 == res) ? "PASSED" : "FAILED");

    // strong_check is in util_quda.cpp.  It generates stdout
    // by calling printSpinorElement(), which is also in util_quda.cpp.
    // See especially compareSpinor() in util_quda.cpp.
    if (test_type < 2) strong_check(spinorRef, spinorOdd, 
      Nh_5d, inv_param.cpu_prec);
    else strong_check(spinorRef, spinorGPU, Nh_5d, inv_param.cpu_prec);

    exit(0);
  }  

  end();

}

int main(int argc, char **argv) {
  dslashTest();
}

