#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <quda_internal.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>
#include <test_util.h>

#include <string.h>
#include "misc.h"

#include <dirac.h>
#include <staggered_dslash_reference.h>

#include <iostream>
#include <mpicomm.h>
#include "exchange_face.h"
#include <mpi.h>

#define mySpinorSiteSize 6
// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)

extern int verbose;

int test_type = 0;
int device = 0;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

FullGauge gauge;
FullGauge cudaFatLink;
FullGauge cudaLongLink;

cpuColorSpinorField*spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;
void *cpu_fwd_nbr_spinor, *cpu_back_nbr_spinor;

cudaColorSpinorField* tmp;

void *hostGauge[4];
void *fatlink[4], *longlink[4];
void* ghost_fatlink, *ghost_longlink;
double kappa = 1.0;
int parity = 0;
QudaDagType dagger = QUDA_DAG_NO;
int TRANSFER = 0; // include transfer time in the benchmark?
int tdim = 8;
int sdim = 8;
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaPrecision prec = QUDA_SINGLE_PRECISION;
const int LOOPS = 20;


Dirac* dirac;
int X[4];
static void
init(void)
{    

  initQuda(device);

  int Vs = sdim*sdim*sdim;
  int Vsh = Vs/2;
  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();
  
  gaugeParam.X[0] = X[0] = sdim;
  gaugeParam.X[1] = X[1] = sdim;
  gaugeParam.X[2] = X[2] = sdim;
  gaugeParam.X[3] = X[3] = tdim;

  setDims(gaugeParam.X);

  gaugeParam.blockDim = 64;

  gaugeParam.cpu_prec = cpu_prec;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;
    
  gaugeParam.anisotropy = 2.3;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.gaugeGiB =0;
    
  // inv_param.gaugeParam = &gaugeParam;
  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = prec;
  if (test_type == 2) inv_param.dirac_order = QUDA_DIRAC_ORDER;
  else inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.kappa = kappa;
  inv_param.solver_type = QUDA_MAT_SOLUTION;
  inv_param.dslash_type = QUDA_STAGGERED_DSLASH;
        
  ColorSpinorParam csParam;
  csParam.fieldType = QUDA_CPU_FIELD;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  for(int d = 0;d < 4; d++) {
    csParam.x[d] = gaugeParam.X[d];
  }
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  if (test_type < 2){
    csParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
    csParam.x[0] /= 2;
  }else{
    csParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;	
  }

  csParam.subsetOrder = QUDA_EVEN_ODD_SUBSET_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_ORDER;
  csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;
  csParam.create = QUDA_ZERO_CREATE;    

  inv_param.sp_pad = sdim*sdim*sdim/2;
  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.fieldSubset = QUDA_FULL_FIELD_SUBSET;
  csParam.x[0] = gaugeParam.X[0];
    
  PRINTF("Randomizing fields ...\n");
    
  spinor->Source(QUDA_RANDOM_SOURCE);
    
  //create ghost spinors
  cpu_fwd_nbr_spinor = malloc(Vsh* mySpinorSiteSize *3*sizeof(double));
  cpu_back_nbr_spinor = malloc(Vsh*mySpinorSiteSize *3*sizeof(double));
  if (cpu_fwd_nbr_spinor == NULL || cpu_back_nbr_spinor == NULL){
    PRINTF("ERROR: malloc failed for cpu_fwd_nbr_spinor/cpu_back_nbr_spinor\n");
    exit(1);
  }
  
  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  for (int dir = 0; dir < 4; dir++) {
    fatlink[dir]  = malloc(V*gaugeSiteSize* gSize);
    longlink[dir] = malloc(V*gaugeSiteSize* gSize);
    if (fatlink[dir] == NULL || longlink[dir] == NULL){
      fprintf(stderr, "ERROR: malloc failed for fatlink/longlink\n");
      exit(1);
    }      
  }    

  ghost_fatlink = malloc(Vs*gaugeSiteSize*gSize);
  ghost_longlink = malloc(3*Vs*gaugeSiteSize*gSize);
  if (ghost_fatlink == NULL || ghost_longlink == NULL){
    PRINTF("ERROR: malloc failed for ghost fatlink/longlink\n");
    exit(1);
  }
    
    
  construct_fat_long_gauge_field(fatlink, longlink, 1, gaugeParam.cpu_prec, &gaugeParam);
    
  
  
  exchange_cpu_links(X, fatlink, ghost_fatlink, longlink, ghost_longlink, gaugeParam.cpu_prec);
  

#if 0
  //PRINTF("links are:\n");
  //display_link(fatlink[0], 1, gaugeParam.cpu_prec);
  //display_link(longlink[0], 1, gaugeParam.cpu_prec);
  
  for (int i =0;i < 4 ;i++){
    int dir = 2*i;
    link_sanity_check(fatlink[i], V, gaugeParam.cpu_prec, dir, &gaugeParam);
    link_sanity_check(longlink[i], V, gaugeParam.cpu_prec, dir, &gaugeParam);
  }

  //PRINTF("spinors are:\n");  
  //display_spinor(spinor, 10, inv_param.cpu_prec);
#endif


  int num_faces =1;
  gaugeParam.ga_pad = sdim*sdim*sdim/2;    
  gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  loadGaugeQuda_general_mg(fatlink, ghost_fatlink, &gaugeParam, &cudaFatLinkPrecise, &cudaFatLinkSloppy, num_faces);
    
  num_faces =3;
  gaugeParam.ga_pad = 3*sdim*sdim*sdim/2;    
  gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = link_recon;
  loadGaugeQuda_general_mg(longlink, ghost_longlink, &gaugeParam, &cudaLongLinkPrecise, &cudaLongLinkSloppy, num_faces);
    
  //gauge = cudaFatLinkPrecise;
  cudaFatLink = cudaFatLinkPrecise;
  cudaLongLink = cudaLongLinkPrecise;
    
  PRINTF("Sending fields to GPU..."); fflush(stdout);
    
  if (!TRANSFER) {
	
    csParam.fieldType = QUDA_CUDA_FIELD;
    csParam.fieldOrder = QUDA_FLOAT2_ORDER;
    csParam.basis = QUDA_DEGRAND_ROSSI_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (test_type < 2){
      csParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
      csParam.x[0] /=2;
    }

	
    PRINTF("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);

    PRINTF("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);
	
    PRINTF("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;
	
    cudaThreadSynchronize();
    checkCudaError();
    
    PRINTF("Source CPU = %f, CUDA=%f\n", norm2(*spinor), norm2(*cudaSpinor));
    
    if(test_type == 2){
      csParam.x[0] /=2;
    }
    csParam.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
    tmp = new cudaColorSpinorField(csParam);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param);
    diracParam.fatGauge = &cudaFatLinkPrecise;
    diracParam.longGauge =&cudaLongLinkPrecise;
	
    diracParam.verbose = QUDA_VERBOSE;
    diracParam.tmp1=tmp;
    dirac = Dirac::create(diracParam);
	
  }else{
    PRINTF("ERROR: not suppported\n");
  }
    


  return;
}

static void
end(void) 
{
  for (int dir = 0; dir < 4; dir++) {
    free(fatlink[dir]);
    free(longlink[dir]);
  }
  free(ghost_fatlink);
  free(ghost_longlink);

  if (!TRANSFER){
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
  }
    
  delete spinor;
  delete spinorOut;
  delete spinorRef;
    
  endQuda();
}

double dslashCUDA() {
    
  // execute kernel
  PRINTF("Executing %d kernel loops...", LOOPS);
  fflush(stdout);
  stopwatchStart();

  for (int i = 0; i < LOOPS; i++) {
    switch (test_type) {
    case 0:
      if (TRANSFER){
	//dslashQuda(spinorOdd, spinorEven, &inv_param, parity, dagger);
      }
      else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity, dagger);
      }	   
      break;
    case 1:
      if (TRANSFER){
	//MatPCQuda(spinorOdd, spinorEven, &inv_param, dagger);
      }else {
	dirac->M(*cudaSpinorOut, *cudaSpinor, dagger);
      }
      break;
    case 2:
      if (TRANSFER){
	//MatQuda(spinorGPU, spinor, &inv_param, dagger);
      }
      else {
	dirac->M(*cudaSpinorOut, *cudaSpinor, dagger);
      }
    }
  }
    
  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    PRINTF("with ERROR: %s\n", cudaGetErrorString(stat));
    
  cudaThreadSynchronize();
  double secs = stopwatchReadSeconds() / LOOPS;
  PRINTF("done.\n\n");
    
  return secs;
}

void 
staggeredDslashRef()
{

  // compare to dslash reference implementation
  PRINTF("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:
    /*
      staggered_dslash(spinorRef->v, fatlink, longlink, spinor->v, parity, dagger, 
      inv_param.cpu_prec, gaugeParam.cpu_prec);
    */
      
    staggered_dslash_mg(spinorRef->v, fatlink, longlink, ghost_fatlink, ghost_longlink, 
			spinor->v, cpu_fwd_nbr_spinor, cpu_back_nbr_spinor, parity, dagger, 
			inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  case 1:    
    staggered_matpc(spinorRef->v, fatlink, longlink, spinor->v, kappa, QUDA_MATPC_EVEN_EVEN, dagger, 
		    inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  case 2:
    //mat(spinorRef->v, fatlink, longlink, spinor->v, kappa, dagger, 
    //inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  default:
    PRINTF("Test type not defined\n");
    exit(-1);
  }
    
  PRINTF("done.\n");
    
}

static void 
dslashTest() 
{
  
  init();
  
  staggeredDslashRef();
  
  double secs = dslashCUDA();
    
  if (!TRANSFER) {
    *spinorOut = *cudaSpinorOut;
  }
    
  PRINTF("%fms per loop\n", 1000*secs);
    
  unsigned long long flops = dirac->Flops()/LOOPS;
  int link_floats = 8*gaugeParam.packed_size+8*18;
  int spinor_floats = 8*6*2 + 6;
  int link_float_size = 0;
  int spinor_float_size = 0;
    
  link_floats = test_type? (2*link_floats): link_floats;
  spinor_floats = test_type? (2*spinor_floats): spinor_floats;
    
  int bytes_for_one_site = link_floats* link_float_size + spinor_floats * spinor_float_size;
    
  PRINTF("GFLOPS = %f\n", 1.0e-9*flops/secs);
  PRINTF("GiB/s = %f\n\n", Vh*bytes_for_one_site/(secs*(1<<30)));
  
  
  PRINTF("Results: CPU = %f, Cuda=%f, CPU-CUDA=%f\n", 
	 norm2(*spinorRef), norm2(*cudaSpinorOut),norm2(*spinorOut));
  
  cpuColorSpinorField::Compare(*spinorRef, *spinorOut);	
  
  if(comm_rank() == 0 || verbose){
    PRINTF("Output spinor:\n");
    spinorOut->PrintVector(0);
    
    PRINTF("Ref spinor:\n");
    spinorRef->PrintVector(0);
  }
  end();
  
}


static void
display_test_info()
{
  PRINTF("running the following test:\n");
 
  PRINTF("prec recon   test_type     dagger   S_dim     T_dimension\n");
  PRINTF("%s   %s       %d           %d       %d        %d \n", 
	 get_prec_str(prec), get_recon_str(link_recon), 
	 test_type, dagger, sdim, tdim);
  return ;
    
}

static void
usage(char** argv )
{
  PRINTF("Usage: %s <args>\n", argv[0]);
  PRINTF("--prec <double/single/half> \t Precision in GPU\n"); 
  PRINTF("--recon <8/12> \t\t\t Long link reconstruction type\n"); 
  PRINTF("--type <0/1/2> \t\t\t Test type\n"); 
  PRINTF("--dagger \t\t\t Set the dagger to 1\n"); 
  PRINTF("--tdim \t\t\t\t Set T dimention size(default 24)\n");     
  PRINTF("--sdim \t\t\t\t Set space dimention size\n"); 
  PRINTF("--help \t\t\t\t Print out this message\n"); 
  exit(1);
  return ;
}

int 
main(int argc, char **argv) 
{

  MPI_Init (&argc, &argv);
  comm_init();
  int i;
  for (i =1;i < argc; i++){
	
    if( strcmp(argv[i], "--help")== 0){
      usage(argv);
    }
    if( strcmp(argv[i], "--device") == 0){
      if (i+1 >= argc){
        usage(argv);
      }
      device =  atoi(argv[i+1]);
      if (device < 0){
        fprintf(stderr, "Error: invalid device number(%d)\n", device);
        exit(1);
      }
      i++;
      continue;
    }	

    if( strcmp(argv[i], "--prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      cpu_prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--recon") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      link_recon =  get_recon(argv[i+1]);
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--type") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      test_type =  atoi(argv[i+1]);
      if (test_type < 0 || test_type > 2){
	fprintf(stderr, "Error: invalid test type\n");
	exit(1);
      }
      i++;
      continue;	    
    }

    if( strcmp(argv[i], "--tdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      tdim =  atoi(argv[i+1]);
      if (tdim < 0 || tdim > 128){
	fprintf(stderr, "Error: invalid t dimention\n");
	exit(1);
      }
      i++;
      continue;	    
    }

    if( strcmp(argv[i], "--sdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      sdim =  atoi(argv[i+1]);
      if (sdim < 0 || sdim > 128){
	fprintf(stderr, "Error: invalid S dimention\n");
	exit(1);
      }
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--dagger") == 0){
      dagger = QUDA_DAG_YES;
      continue;	    
    }	
    
    if( strcmp(argv[i], "--verbose") == 0){
      verbose = 1;
      continue;	    
    }


    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }
  display_test_info();
  
  dslashTest();
  
  comm_exit(0);
}
