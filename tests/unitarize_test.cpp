#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <quda_internal.h>

#include <quda.h>
#include <gauge_quda.h>
#include <dslash_quda.h>
#include <unitarize_quda.h>

#include <llfat_quda.h>

#include <test_util.h>
#include <unitarize_reference.h>
#include <llfat_reference.h>
#include "misc.h"


#include "fat_force_quda.h"

#ifdef MULTI_GPU
#include "face_quda.h"
#include "mpicomm.h"
#include <mpi.h>
#endif


FullGauge cudaInLink;
FullGauge cudaFatLink;
FullGauge cudaOutLink;

FullStaple cudaStaple;
FullStaple cudaStaple1;
QudaGaugeParam gaugeParam;
void *inlink[4], *fatlink[4]; // outlink is used to store the GPU output; reflink stores the CPU output
void *templink, *outlink, *reflink;

#ifdef MULTI_GPU
void *ghost_inlink[4];
void *ghost_inlink_diag[16];
#endif


int compare_results = 1;
int check_unitarity = 1;

extern void initDslashCuda(FullGauge gauge);

#define DIM 24

int device = 0;
int ODD_BIT = 1;


// A struct I added to tidy the code a little bit
struct DimensionHolder{
  int dim[4];
  int vol;
  int half_vol;
  int slice_vol[4];
  int half_slice_vol[4];
};




extern int xdim, ydim, zdim, tdim;  // local lattice dimensions -> initialised in "test_util.cpp"
extern int gridsize_from_cmdline[]; // Four-dimensional array initialised in "test_util.cpp" - it holds the grid dimensions


extern QudaReconstructType link_recon; // link_recon is a global variable defined in "test_util.cpp"
extern QudaPrecision prec; // This is the gpu link precision, I believe

//QudaPrecision gpu_link_prec = QUDA_SINGLE_PRECISION;
//QudaPrecision cpu_link_prec = QUDA_SINGLE_PRECISION;
QudaPrecision gpu_link_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cpu_link_prec = QUDA_DOUBLE_PRECISION;
//QudaLinkType  smearing_scheme = QUDA_ASQTAD_FAT_LINKS;

typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct { dcomplex e[3][3]; } d3x3_matrix;


int V;    // Yuck!
int Vh;   //
int Z[4]; // These global variables are needed in other functions

void setDims(DimensionHolder* dh, const int* const X) {

  dh->vol=1;
  for(int d=0; d<4; ++d){ 
    dh->vol *= X[d];
    dh->dim[d] = X[d];
  }
  dh->half_vol = dh->vol/2;

  dh->slice_vol[0] = X[1]*X[2]*X[3];
  dh->slice_vol[1] = X[0]*X[2]*X[3];
  dh->slice_vol[2] = X[0]*X[1]*X[3];
  dh->slice_vol[3] = X[0]*X[1]*X[2];

  for(int d=0; d<4; ++d){ dh->half_slice_vol[d] = dh->slice_vol[d]/2; }
   
  // Set those nasty global variables
  V  = dh->vol;
  Vh = dh->half_vol;
  for(int d=0; d<4; ++d){ Z[d] = dh->dim[d]; }

  return;
}


// allocate memory for, and zero elements of the ghost arrays
static void allocate_zero_ghost_arrays(void* ghost[], void* ghost_diag[], const DimensionHolder& dh, const QudaGaugeParam & p){
  const size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for(int d=0; d<4; ++d){
    ghost[d] = malloc(8*dh.slice_vol[d]*gaugeSiteSize*gSize);
    if(ghost[d] == NULL){
      printf("ERROR: malloc failed for ghost[%d] \n", d);
      exit(1);
    }
  }

  for(int nu=0; nu<4; ++nu){
    for(int mu=0; mu<4; ++mu){
      if(nu==mu){
        ghost_diag[nu*4+mu] = NULL;
      }else{
        int dir1, dir2;

        for(dir1=0; dir1<4; ++dir1){
          if(dir1 != nu  && dir1 != mu) break;
        }

        for(dir2=0; dir2<4; ++dir2){
          if(dir2 != nu && dir2 != mu && dir2 != dir1) break;
        }                                                                                                                               

        ghost_diag[nu*4+mu] = malloc(dh.dim[dir1]*dh.dim[dir2]*gaugeSiteSize*gSize);
        if(ghost_diag[nu*4+mu] == NULL){
          errorQuda("malloc failed for ghost_diag\n");
        }
        memset(ghost_diag[nu*4+mu], 0, dh.dim[dir1]*dh.dim[dir2]*gaugeSiteSize*gSize);
      }
    }
  }
  return;
}


static int get_max_2d_vol(const int* const dim){
  int max_vol_2d = 0;
  int i, j, vol_2d;
  for(i=0; i<4; ++i){
    for(j=i+1; j<4; ++j){
      vol_2d = dim[i]*dim[j];
      if(vol_2d > max_vol_2d){
        max_vol_2d = vol_2d;
      }
    }
  } // end loop over directions
  return max_vol_2d;
}






static void unitarize_init(void){
  initQuda(device);

  DimensionHolder dh; 
  // space time dimensions 
  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  std::cout << "xdim = " << xdim << std::endl;
  std::cout << "ydim = " << ydim << std::endl;
  std::cout << "zdim = " << zdim << std::endl;
  std::cout << "tdim = " << tdim << std::endl;

  setDims(&dh, gaugeParam.X);
    
  gaugeParam.cpu_prec  = cpu_link_prec;
  gaugeParam.cuda_prec = gpu_link_prec;
        
  const size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
   
  // May need to rearrange inlink 
  // In Guochun's updated code sitelink is a two dimensional array 
  for(int dir=0; dir<4; ++dir){
    inlink[dir] = malloc(dh.vol*gaugeSiteSize*gSize);
    if (inlink[dir] == NULL) {
      fprintf(stderr, "ERROR: malloc failed for inlink[%d]\n",dir);
    }
  }

  for(int dir=0; dir<4; ++dir){
    fatlink[dir] = malloc(dh.vol*gaugeSiteSize*gSize);
    if (fatlink[dir] == NULL) {
      fprintf(stderr, "ERROR: malloc failed for fatlink[%d]\n",dir);
    }
  }


   
  // Finally, allocate a one-dimensional array for the temporary link
  outlink  = malloc(4*dh.vol*gaugeSiteSize*gSize);
  reflink  = malloc(4*dh.vol*gaugeSiteSize*gSize);
  templink = malloc(4*dh.vol*gaugeSiteSize*gSize);
 

#ifdef MULTI_GPU
  allocate_zero_ghost_arrays(ghost_inlink, ghost_inlink_diag, dh, gaugeParam);
#endif

    
  createSiteLinkCPU(inlink, gaugeParam.cpu_prec, 1);

  strong_check_link(inlink, inlink, V, gaugeParam.cpu_prec);

  gaugeParam.reconstruct = link_recon;

#ifdef MULTI_GPU
  const int Vsh_sum = dh.half_slice_vol[0] + dh.half_slice_vol[1] + dh.half_slice_vol[2] + dh.half_slice_vol[3];
  const int Vh_2d_max = get_max_2d_vol(dh.dim)/2;

  printf("Vh_2d_max = %d\n", Vh_2d_max);

  gaugeParam.site_ga_pad = gaugeParam.ga_pad = 3*Vsh_sum + 4*Vh_2d_max;
  createLinkQuda(&cudaInLink, &gaugeParam);
  loadLinkToGPU(cudaInLink, inlink, &gaugeParam);
#else
  gaugeParam.site_ga_pad = gaugeParam.ga_pad = dh.half_slice_vol[3];
  createLinkQuda(&cudaInLink, &gaugeParam);
  fprintf(stderr, "ERROR: single core currently not supported\n");
  fprintf("Check back soon\n");
  loadLinkToGPU(cudaInLink, inlink, NULL, NULL, &gaugeParam);
#endif  
  gaugeParam.staple_pad = 3*Vsh_sum;
  createStapleQuda(&cudaStaple, &gaugeParam);
  createStapleQuda(&cudaStaple1, &gaugeParam);

  gaugeParam.llfat_ga_pad = gaugeParam.ga_pad = dh.half_slice_vol[3];
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;

  createLinkQuda(&cudaFatLink, &gaugeParam);
  createLinkQuda(&cudaOutLink, &gaugeParam);

  initDslashConstants(cudaInLink, 0);

  return;
}



void 
unitarize_end() 
{
  for(int dir=0; dir<4; ++dir) free(inlink[dir]); 
  for(int dir=0; dir<4; ++dir) free(fatlink[dir]); 

  // ...and 
  free(outlink);
  free(reflink);
  free(templink);

  for(int dir=0; dir<4; ++dir) free(ghost_inlink[dir]);
  
  for(int i=0; i<16; ++i){
    if(ghost_inlink_diag[i] != NULL){
      free(ghost_inlink_diag[i]);
	  }
  }

  freeLinkQuda(&cudaInLink);
  freeLinkQuda(&cudaFatLink);
  freeLinkQuda(&cudaOutLink);
  freeStapleQuda(&cudaStaple);
  freeStapleQuda(&cudaStaple1);

  endQuda();
}

template<class Float>
void set_unit_link(Float *link){

  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j){
      int re_index = 2*(i*3+j);
      int im_index = re_index + 1;
      link[re_index] = link[im_index] = 0.0;
      if(i == j) link[re_index] = 1.0;
    }
  }
  return;
}


static void set_unit_field(void *link, int len, int precision){
  if(precision == QUDA_DOUBLE_PRECISION){
    double* mylink = (double*)link;
    for(int i=0; i<len; ++i){
      for(int dir=XUP; dir<=TUP; ++dir){
        set_unit_link(mylink + gaugeSiteSize*(4*i+dir));
      }
    }
  }else{
	  printf("In set_unit_field\n");
    printf("Single precision not yet supported\n");
  }

  return;
}

static void unitarize_test(void) 
{
  unitarize_init();
  unitarize_init_cuda(&gaugeParam);


  gaugeParam.ga_pad = gaugeParam.llfat_ga_pad;

  float act_path_coeff_1[6];
  double act_path_coeff_2[6];
  
  for(int i=0; i<6; i++){
    act_path_coeff_1[i] = 0.;
    act_path_coeff_2[i] = 0.;
  }

  act_path_coeff_1[0] = 1./8.;
  act_path_coeff_1[2] = 1./16.;
  act_path_coeff_1[3] = 1./64.;
  act_path_coeff_1[4] = 1./384.;

  act_path_coeff_2[0] = 1./8.;
  act_path_coeff_2[2] = 1./16.;
  act_path_coeff_2[3] = 1./64.;
  act_path_coeff_2[4] = 1./384.;



  void* act_path_coeff;    
  if(gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
    act_path_coeff = act_path_coeff_2;
  }else{
    act_path_coeff = act_path_coeff_1;	
  }
  double secs;
  struct timeval t0, t1;

  llfat_cuda(cudaFatLink, cudaInLink, cudaStaple, cudaStaple1, &gaugeParam, act_path_coeff_2);
  cudaThreadSynchronize();
 
  gettimeofday(&t0, NULL);  
  storeLinkToCPU(templink, &cudaFatLink, &gaugeParam);
  unitarize_reference(reflink, templink, gaugeParam.cpu_prec);
  gettimeofday(&t1, NULL);
  secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
  printf("CPU unitarization time = %.2f ms\n", secs*1000);


  
  gettimeofday(&t0, NULL);
  unitarize_cuda_si(cudaOutLink, cudaFatLink, &gaugeParam, 18);  
  // unitarize_cuda_hc(cudaOutLink, cudaFatLink, &gaugeParam); 
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
  printf("GPU unitarization time = %.2f ms\n", secs*1000);


  

  gettimeofday(&t0, NULL);
  storeLinkToCPU(outlink, &cudaOutLink, &gaugeParam);    
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  secs = t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
  printf("Link Store time = %.2f ms\n", secs*1000);

  

  if(compare_results){
   // strong_check_link(reflink, outlink, 4*V, gaugeParam.cpu_prec);
    int res = compare_floats(reflink, outlink, 4*V*gaugeSiteSize, 1e-3, gaugeParam.cpu_prec);
    printf("Comparison test %s\n", (1==res) ? "PASSED" : "FAILED");
  }


  int res;
  if(check_unitarity){
		res = check_field_unitarity(outlink, V, gaugeParam.cpu_prec); 
    printf("GPU unitarity test %s\n", (1==res) ? "PASSED" : "FAILED");
  }



  unitarize_end();
    
//  if (res == 0){//failed
//    printf("\n");
//    printf("Warning: your test failed. \n");
//    printf("	Did you use --check?\n");
//  }
}            




int main(int argc, char *argv[]) 
{
  // default to 18 reconstruct, 8^3 x 8
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim=ydim=zdim=tdim=16;
 
  prec = cpu_link_prec; // Need to rewrite this! 
  int i;
  for (i=1; i<argc; ++i){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
   
	
    if( strcmp(argv[i], "--help")== 0){
      printf("usage to follow\n");	
    }
	
    if( strcmp(argv[i], "--compare") == 0){
      compare_results=1;
      continue;	    
    }

    if( strcmp(argv[i], "--check") == 0){
      compare_results=1;
      check_unitarity=1;
      continue;
    }	

  }

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);    
  unitarize_test();

  endCommsQuda();
    
  return 0;
}
