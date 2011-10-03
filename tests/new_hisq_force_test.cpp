#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <quda.h>
#include <dslash_quda.h>
#include "test_util.h"
#include "gauge_quda.h"
#include "misc.h"
#include "fat_force_quda.h"
#include "hisq_force_reference.h"
#include "hisq_force_quda.h"
#include "hw_quda.h"
#include "hisq_force_utils.h"
#include <sys/time.h>


using namespace hisq::fermion_force;
static FullGauge cudaSiteLink;
static FullGauge cudaMomMatrix;
static FullMom  cudaMom;
static FullHw cudaHw;
static QudaGaugeParam gaugeParam;
static QudaGaugeParam momMatrixParam;

static void* siteLink;
static void* mom;
static void* refMom;
static void* hw;
static void* oprod; // outer product of half wilson vectors
static int X[4];

static void* cudaOprodEven;
static void* cudaOprodOdd;
static void* return_oprod;

static FullOprod cudaOprod;

extern int gridsize_from_cmdline;

int verify_results = 0;

#ifdef __cplusplus
extern "C" {
#endif

extern void initDslashCuda(FullGauge gauge);
extern void initDslashConstants(const FullGauge gauge, const int sp_stride);

#ifdef __cplusplus
}
#endif

int ODD_BIT = 1;
extern int xdim, ydim, zdim, tdim;

extern QudaReconstructType link_recon;
QudaPrecision link_prec = QUDA_SINGLE_PRECISION;
extern QudaPrecision prec;
QudaPrecision hw_prec = QUDA_SINGLE_PRECISION;
QudaPrecision mom_prec = QUDA_SINGLE_PRECISION;

QudaPrecision cpu_hw_prec = QUDA_SINGLE_PRECISION;

typedef struct {
  double real;
  double imag;
} dcomplex;


typedef struct { dcomplex e[3][3]; } dsu3_matrix;


int Z[4];
int V;
int Vh;

void 
setDims(int *X){
  V=1;
  for(int dir=0; dir<4; ++dir){
    V *= X[dir];
    Z[dir] = X[dir];
  }
  Vh = V/2;
  return;
}



static void 
hisq_force_init()
{ 
  initQuda(device);

  X[0] = gaugeParam.X[0] = xdim;
  X[1] = gaugeParam.X[1] = ydim;
  X[2] = gaugeParam.X[2] = zdim;
  X[3] = gaugeParam.X[3] = tdim;

  for(int dir=0; dir<4; ++dir){ 
    momMatrixParam.X[dir] = X[dir];
  }


  setDims(gaugeParam.X);

  gaugeParam.cpu_prec = link_prec;
  gaugeParam.cuda_prec = link_prec;
  gaugeParam.reconstruct = link_recon;

  momMatrixParam.cpu_prec = link_prec;
  momMatrixParam.cuda_prec = link_prec;
  momMatrixParam.reconstruct = QUDA_RECONSTRUCT_NO;

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  siteLink = malloc(4*V*gaugeSiteSize*gSize);
  if(siteLink == NULL){
    fprintf(stderr, "ERROR: malloc failed for sitelink\n");
    exit(1);
  }

  void* siteLink_2d[4];
  for(int dir=0; dir<4; ++dir){
    siteLink_2d[dir] = ((char*)siteLink + dir*V*gaugeSiteSize*gSize);
  }

  createSiteLinkCPU(siteLink_2d, gaugeParam.cpu_prec, 0);

  mom = malloc(4*V*momSiteSize*gSize);
  if(mom == NULL){
    fprintf(stderr, "ERROR: malloc failed for mom\n");
    exit(1);
  }
  createMomCPU(mom,mom_prec);
  memset(mom,0, 4*V*momSiteSize*gSize);

  refMom = malloc(4*V*momSiteSize*gSize);
  if (refMom == NULL){
    fprintf(stderr, "ERROR: malloc failed for refMom\n");
    exit(1);
  }
  memcpy(refMom, mom, 4*V*momSiteSize*gSize);


  hw = malloc(V*hwSiteSize*gSize);
  if (hw == NULL){
    fprintf(stderr, "ERROR: malloc failed for hw\n");
    exit(1);
  }
  createHwCPU2(hw, hw_prec);

  createLinkQuda(&cudaSiteLink, &gaugeParam);
  createMomQuda(&cudaMom, &gaugeParam);
  cudaHw = createHwQuda(X, hw_prec);


  oprod = malloc(8*V*gaugeSiteSize*sizeof(float));
  computeHisqOuterProduct(hw, oprod);
  allocateOprodFields(&cudaOprodEven, &cudaOprodOdd, Vh);
  cudaOprod = createOprodQuda(X, hw_prec);


  return;
} // end hisq_force_init  


static void 
hisq_force_end()
{ 
  free(siteLink);
  free(mom);
  free(refMom);
  free(hw);

  freeLinkQuda(&cudaSiteLink);
  freeLinkQuda(&cudaMomMatrix);
  freeMomQuda(&cudaMom);
  // I need to free cudaOprod
}


static void 
checkDifference(float* a, float* b, int total_size){
  float delta = 0.001;
  for(int i=0; i<total_size; ++i){
    if(fabs(a[i]-b[i])>delta){
      printf("Mismatch at i = %d\n", i);
      exit(1);
    }
    if(a[i] == 0.){
		  printf("Zero element at i = %d\n", i);
    }
  }
  return;
}


static void 
hisq_force_test(void)
{
  hisq_force_init();
  initDslashConstants(cudaSiteLink, Vh);
  hisq_force_init_cuda(&gaugeParam);

  float eps= 0.02;
  float weight1 =1.0;
  float weight2 =0.0;
  float act_path_coeff[6];

  act_path_coeff[0] = 0.625000;
  act_path_coeff[1] = -0.058479;
  act_path_coeff[2] = -0.087719;
  act_path_coeff[3] = 0.030778;
  act_path_coeff[4] = -0.007200;
  act_path_coeff[5] = -0.123113;

  loadMomToGPU(cudaMom, mom, &gaugeParam);
  loadLinkToGPU_gf(cudaSiteLink, siteLink, &gaugeParam);
  loadHwToGPU(cudaHw, hw, cpu_hw_prec);

   // load the Oprod
   if(verify_result){
     color_matrix_hisq_force_reference(eps,weight1, act_path_coeff, return_oprod, siteLink, refMom);
   }

   hisq_force_cuda(eps, weight1, weight2, act_path_coeff, cudaOprod, cudaSiteLink, cudaMom, cudaMomMatrix, &gaugeParam);

   return; 
}



static void
usage(char** argv )
{
  printf("Usage: %s <args>\n", argv[0]);
  printf("  --device <dev_id>               Set which device to run on\n");
  printf("  --gprec <double/single/half>    Link precision\n");
  printf("  --recon <8/12>                  Link reconstruction type\n");
  printf("  --sdim <n>                      Set spacial dimention\n");
  printf("  --tdim                          Set T dimention size(default 24)\n");
  printf("  --sdim                          Set spalce dimention size(default 16)\n");
  printf("  --verify                        Verify the GPU results using CPU results\n");
  printf("  --help                          Print out this message\n");
  exit(1);
  return ;
}


