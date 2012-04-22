#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <test_util.h>
#include <twisted_mass_dslash_reference.h>

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
const int test_type = 0;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const QudaDagType dagger = QUDA_DAG_YES;     // apply Dslash or Dslash dagger?
const int transfer = 0; // include transfer time in the benchmark?

const int loops = 1;
const int Nf = 2;

QudaPrecision cpu_prec  = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec = QUDA_SINGLE_PRECISION;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

cpuColorSpinorField  *spinor,   *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp=0, *tmp2=0;

void *hostGauge[4];

Dirac *dirac;

void init() {

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = 16;
  gauge_param.X[1] = 16;
  gauge_param.X[2] = 16;
  gauge_param.X[3] = 32;
  setDims(gauge_param.X);

  gauge_param.anisotropy = 1.0;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
  gauge_param.reconstruct_sloppy = gauge_param.reconstruct;
  gauge_param.cuda_prec_sloppy = gauge_param.cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  inv_param.kappa = 0.1;
  inv_param.mu = 0.5;
  inv_param.epsilon = 0.01;
  inv_param.twist_flavor = QUDA_TWIST_DUPLET;

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = dagger;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;

  gauge_param.ga_pad = 0;
  inv_param.sp_pad   = 0;
  inv_param.cl_pad   = 0;

  //gauge_param.ga_pad = 24*24*24;
  //inv_param.sp_pad = 24*24*24;
  //inv_param.cl_pad = 24*24*24;

  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (test_type == 2) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
  }

  inv_param.dslash_type = QUDA_TWISTED_MASS_DSLASH;

  inv_param.verbosity = QUDA_VERBOSE;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.twistFlavor = inv_param.twist_flavor;
///NB!  
  csParam.nDim = 5;
  
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
///NB!
  csParam.x[4] = Nf;
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  
  if (test_type < 2) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }    
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  
  spinor    = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  construct_gauge_field(hostGauge, 0, gauge_param.cpu_prec, &gauge_param);
  
  spinor->Source(QUDA_RANDOM_SOURCE);
  
  printfQuda("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);

  printfQuda("Sending gauge field to GPU\n");

  loadGaugeQuda(hostGauge, &gauge_param);
  //gauge = cudaGaugePrecise;

  if (!transfer) {
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if (test_type < 2) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor    = new cudaColorSpinorField(csParam);
  
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    if (test_type == 2) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");

printfQuda("\n\nCUDA spinor info\n\n");    
std::cout<<*cudaSpinor;    
printfQuda("\nCPU spinor info\n\n");    
std::cout<<*spinor;        
    
    *cudaSpinor = *spinor;

    std::cout << "Source: CPU = " << norm2(*spinor) << ", CUDA = " << 
      norm2(*cudaSpinor) << std::endl;

      
    bool pc = (test_type != 2);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.verbose = QUDA_VERBOSE;
    diracParam.tmp1 = tmp;
    diracParam.tmp2 = tmp2;
    
    dirac = Dirac::create(diracParam);
  } else {
    std::cout << "Source: CPU = " << norm2(*spinor) << std::endl;
  }
    
}

void end() {
  if (!transfer) {
    delete dirac;
    delete cudaSpinor;    
    delete cudaSpinorOut;    
    delete tmp;
    delete tmp2;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;
  
  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  endQuda();
}

// execute kernel
double ndegDslashCUDA() {

  printfQuda("Executing %d kernel loops...\n", loops);
  fflush(stdout);
  
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);
  
  for (int i = 0; i < loops; i++) {
    switch (test_type) {
    case 0:
      if (transfer) {
	dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
      } else if(inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH){
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }
      break;
    case 1:
    {
        printfQuda("\ncompute Mdag\n");//See CPU version!
        dirac->Mdag(*cudaSpinorOut, *cudaSpinor);
    }
      break;
    case 2:
      if (transfer) {
	//MatQuda(spinorOut->v, spinor->v, &inv_param);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
      break;
    }
  }
    
  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000; //stopwatchReadSeconds();

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  printf("done.\n\n");

  return secs;
}

void ndegDslashRef() {

  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  int flv_offset = 12*spinorRef->Volume();//24*Vh (whole parity volume 2*24*Vh)
  switch (test_type) {
  case 0:
  {
    //change this!
    printf("\nCPU flavor volume: %d\n", flv_offset);
    void *ref1 = spinorRef->V();
    void *ref2 = cpu_prec == sizeof(double) ? (void*)((double*)ref1 + flv_offset): (void*)((float*)ref1 + flv_offset);
    
    void *flv1 = spinor->V();
    void *flv2 = cpu_prec == sizeof(double) ? (void*)((double*)flv1 + flv_offset): (void*)((float*)flv1 + flv_offset);
    
    ndeg_dslash(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon,
	   parity, dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
  }
    break;
  case 1: //NOTE !dagger   
    printf("Test type is disabled\n");    
    //ndeg_matpc(spinorRef1->v, spinorRef2->v, hostGauge, spinor1->v, spinor2->v, inv_param.kappa, inv_param.mu, inv_param.epsilon,
	  //inv_param.matpc_type, !dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  case 2:
    printf("Test type is disabled\n");        
    //mat(spinorRef->v, hostGauge, spinor->v, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	//dagger, inv_param.cpu_prec, gauge_param.cpu_prec);
    break;
  default:
    printf("Test type not defined\n");
    exit(-1);
  }

  printf("done.\n");
    
}

int main(int argc, char **argv)
{
  init();

  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  float sharedKB = 0;//(float)dslashCudaSharedBytes(inv_param.cuda_prec) / (1 << 10);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  printf("Shared mem: %.3f KB\n", sharedKB);
  
  int attempts = 1;
  ndegDslashRef();
  for (int i=0; i<attempts; i++) {
    
    double secs = ndegDslashCUDA();

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printf("%fms per loop\n", 1000*secs);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int floats = test_type ? 2*(7*24+8*gauge_param.packed_size+24)+24 : 7*24+8*gauge_param.packed_size+24;

    printf("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printf("GiB/s = %f\n\n", Vh*floats*sizeof(float)/((secs/loops)*(1<<30)));
    
    if (!transfer) {
      std::cout << "Results: CPU = " << norm2(*spinorRef) <<  
	", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    } else {
      std::cout << "Result: CPU = " << norm2(*spinorRef) << ", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    }
    
    cpuColorSpinorField::Compare(*spinorRef, *spinorOut);
  } 

//Print CPU vs GPU:
for(int i = 0; i < 1; i++)
{
printf("\nCPU spinor:\n");
spinorRef->PrintVector(i);
printf("\nGPU spinor:\n");
spinorOut->PrintVector(i);
}     
  end();
}
