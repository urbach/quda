#include <read_gauge.h>
#include <gauge_quda.h>

#include "hisq_force_quda.h"
#include "force_common.h"
#include "hw_quda.h"
#include "hisq_force_macros.h"


namespace hisq {
  namespace fermion_force {


#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_SINGLE



#define LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)

#define FF_SITE_MATRIX_LOAD_TEX 1

#if (FF_SITE_MATRIX_LOAD_TEX == 1)
#define linkEvenTex siteLink0TexSingle_recon
#define linkOddTex siteLink1TexSingle_recon
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(src##Tex, dir, idx, var)
#else
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)
#endif


#define LOAD_ARRAY_12_SINGLE_TEX(gauge, dir, idx, var)do{                     \
      var[0] = tex1Dfetch(gauge, idx + dir*Vhx3);                         \
      var[1] = tex1Dfetch(gauge, idx + dir*Vhx3 + Vh);                    \
      var[2] = tex1Dfetch(gauge, idx + dir*Vhx3 + Vhx2);                  \
    }while(0)

#define FF_LOAD_ARRAY(src, dir, idx, var) LOAD_ARRAY_12_SINGLE_TEX(src##Tex, dir, idx, var)    


//void load_array_12_from_texture(float4 array[4], int dir, int idx) 

// Need to compute the neighbouring index
// Can really clean the code up
// I need to force the compiler to inline these functions

__forceinline__ __device__ 
void computeNewFullIndexPlusUpdate(int dir, int idx, int new_x[4], int & new_index){
  switch(dir){
    case 0:
      new_index = ((new_x[0]==X1m1)?idx-X1m1:idx+1);		
      new_x[0] = (new_x[0]==X1m1)?0:new_x[0]+1;                       
    break;

    case 1:                                                        
      new_index = ( (new_x[1]==X2m1)?idx-X2X1mX1:idx+X1);		
      new_x[1] = (new_x[1]==X2m1)?0:new_x[1]+1;    
    break;     

    case 2:                                                         
      new_index = ( (new_x[2]==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	
      new_x[2] = (new_x[2]==X3m1)?0:new_x[2]+1;                         
    break;                              

    case 3:                                                         
      new_index = ( (new_x[3]==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); 
      new_x[3] = (new_x[3]==X4m1)?0:new_x[3]+1;                        
    break;      

    case 4:    
      new_index = ( (new_x[0]==0)?idx+X1m1:idx-1);                 
      new_x[0] = (new_x[0]==0)?X1m1:new_x[0] - 1;                       
    break;                                                      

    case 5:                                                         
      new_index = ( (new_x[1]==0)?idx+X2X1mX1:idx-X1);              
      new_x[1] = (new_x[1]==0)?X2m1:new_x[1] - 1;                       
    break;                                                      
  }
  return;
}


#define FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mydir, idx, new_idx) do {	\
  switch(mydir){                                                  \
    case 0:                                                         \
                                                                    new_idx = ( (new_x[0]==X1m1)?idx-X1m1:idx+1);			\
    new_x[0] = (new_x[0]==X1m1)?0:new_x[0]+1;                         \
    break;                                                      \
    case 1:                                                         \
                                                                    new_idx = ( (new_x[1]==X2m1)?idx-X2X1mX1:idx+X1);		\
    new_x[1] = (new_x[1]==X2m1)?0:new_x[1]+1;                         \
    break;                                                      \
    case 2:                                                         \
                                                                    new_idx = ( (new_x[2]==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	\
    new_x[2] = (new_x[2]==X3m1)?0:new_x[2]+1;                         \
    break;                                                      \
    case 3:                                                         \
                                                                    new_idx = ( (new_x[3]==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); \
    new_x[3] = (new_x[3]==X4m1)?0:new_x[3]+1;                         \
    break;                                                      \
  }                                                               \
}while(0)

#define FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mydir, idx, new_idx) do {	\
  switch(mydir){                                                  \
    case 0:                                                         \
                                                                    new_idx = ( (new_x[0]==0)?idx+X1m1:idx-1);			\
    new_x[0] = (new_x[0]==0)?X1m1:new_x[0] - 1;                       \
    break;                                                      \
    case 1:                                                         \
                                                                    new_idx = ( (new_x[1]==0)?idx+X2X1mX1:idx-X1);		\
    new_x[1] = (new_x[1]==0)?X2m1:new_x[1] - 1;                       \
    break;                                                      \
    case 2:                                                         \
                                                                    new_idx = ( (new_x[2]==0)?idx+X3X2X1mX2X1:idx-X2X1);		\
    new_x[2] = (new_x[2]==0)?X3m1:new_x[2] - 1;                       \
    break;                                                      \
    case 3:                                                         \
                                                                    new_idx = ( (new_x[3]==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1);	\
    new_x[3] = (new_x[3]==0)?X4m1:new_x[3] - 1;                       \
    break;                                                      \
  }                                                               \
}while(0)



#define FF_COMPUTE_NEW_FULL_IDX_PLUS(old_x1, old_x2, old_x3, old_x4, idx, mydir, new_idx) do { \
  switch(mydir){                                                  \
    case 0:                                                         \
                                                                    new_idx = ( (old_x1==X1m1)?idx-X1m1:idx+1);			\
    break;                                                      \
    case 1:                                                         \
                                                                    new_idx = ( (old_x2==X2m1)?idx-X2X1mX1:idx+X1);		\
    break;                                                      \
    case 2:                                                         \
                                                                    new_idx = ( (old_x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	\
    break;                                                      \
    case 3:                                                         \
                                                                    new_idx = ( (old_x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); \
    break;                                                      \
  }                                                               \
}while(0)

#define FF_COMPUTE_NEW_FULL_IDX_MINUS(old_x1, old_x2, old_x3, old_x4, idx, mydir, new_idx) do { \
  switch(mydir){                                                  \
    case 0:                                                         \
                                                                    new_idx = ( (old_x1==0)?idx+X1m1:idx-1);			\
    break;                                                      \
    case 1:                                                         \
                                                                    new_idx = ( (old_x2==0)?idx+X2X1mX1:idx-X1);		\
    break;                                                      \
    case 2:                                                         \
                                                                    new_idx = ( (old_x3==0)?idx+X3X2X1mX2X1:idx-X2X1);		\
    break;                                                      \
    case 3:                                                         \
                                                                    new_idx = ( (old_x4==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1);	\
    break;                                                      \
  }                                                               \
}while(0)





//this macro require link_W, link_X and ah variables defined
#define SIMPLE_MAT_FORCE_TO_MOM(mat, mom, idx, dir, temp_mat) do { \
  {                                                             \
  float2 AH0, AH1, AH2, AH3, AH4;                               \
  LOAD_ANTI_HERMITIAN(mom, dir, idx, AH);			\
  UNCOMPRESS_ANTI_HERMITIAN(ah, temp_mat);			\
  SCALAR_MULT_ADD_SU3_MATRIX(temp_mat, mat, 1.0, link_W);	\
  MAKE_ANTI_HERMITIAN(temp_mat, ah);				\
  WRITE_ANTI_HERMITIAN_SINGLE(mom, dir, idx, AH);		\
  }                                                             \
}while(0)






// Struct to determine the coefficient sign at compile time
template<int pos_dir, int odd_lattice>
struct CoeffSign
{
    static const int result = -1;
};

template<>
struct CoeffSign<0,0>
{
    static const int result = 1;
};

template<>
struct CoeffSign<1,1>
{
    static const int result = 1;
};




__device__ void reconstructSign(int* const sign, int dir, int i[4]){
  *sign=1;
  switch(dir){
    case XUP:
      if( (i[3]&1)==1) *sign=1;
    break;

    case YUP:
      if( ((i[3]+i[0])&1) == 1) *sign=1; 
    break;

    case ZUP:
      if( ((i[3]+i[0]+i[1])&1) == 1) *sign=1; 
    break;

    case TUP:
      if(i[3] == X4m1) *sign=1; 
    break;
  }
}



void
hisq_force_init_cuda(QudaGaugeParam* param)
{
  static int fermion_force_init_cuda_flag = 0; 

  if (fermion_force_init_cuda_flag){
    return;
  }
  fermion_force_init_cuda_flag=1;
  init_kernel_cuda(param);    
}




template<int oddBit>
  __global__ void 
do_compute_force_kernel(float4* linkEven, float4* linkOdd,
    float2* momMatrixEven, float2* momMatrixOdd,
    int sig,
    float2* momEven, float2* momOdd)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int link_sign;

  float4 LINK_W[5];
  float2 COLOR_MAT_W[9];
  float2 COLOR_MAT_X[9];

  FF_LOAD_ARRAY(linkEven, sig, sid, LINK_W);
  reconstructSign(&link_sign, sig, x);	
  RECONSTRUCT_LINK_12(sig, sid, link_sign, link_W);

  LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_X);
  MAT_MUL_MAT(link_W, color_mat_X, color_mat_W);

  SIMPLE_MAT_FORCE_TO_MOM(color_mat_W, momEven, sid, sig, link_W);
}





template<int sig_positive, int mu_positive, int oddBit> 
__global__ void
do_middle_link_kernel(const float2 * const tempEven, 
    float2 * const PmuOdd, float2 * const P3Even,
    const float2 * const QprevOdd, 		
    float2 * const QmuEven, 
    int sig, int mu, float coeff,
    float4 * const linkEven, float4 * const linkOdd,
    float2 * const momEven,
    float2 * const momMatrixEven 
    ) 
{		
  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int new_x[4];
  int new_mem_idx;
  int ad_link_sign=1;
  int ab_link_sign=1;
  int bc_link_sign=1;

  float4 LINK_W[5];
  float4 LINK_X[5];
  float4 LINK_Y[5];
  float4 LINK_Z[5];


  float2 COLOR_MAT_W[9];
  float2 COLOR_MAT_Y[9];
  float2 COLOR_MAT_X[9];
  float2 COLOR_MAT_Z[9];

//  float2 AH0, AH1, AH2, AH3, AH4;

  //        A________B
  //    mu   |      |
  // 	    D|      |C
  //	  
  //	  A is the current point (sid)
  int point_b, point_c, point_d;
  int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
  int mymu;

  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(mu_positive){
    mymu =mu;
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
  }else{
    mymu = OPP_DIR(mu);
   // computeNewFullIndexPlusUpdate(OPP_DIR(mu),X,new_x,new_mem_idx);
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(OPP_DIR(mu), X, new_mem_idx);	
  }
  point_d = (new_mem_idx >> 1);
  if (mu_positive){
    ad_link_nbr_idx = point_d;
    reconstructSign(&ad_link_sign, mymu, new_x);
  }else{
    ad_link_nbr_idx = sid;
    reconstructSign(&ad_link_sign, mymu, x);	
  }


  int mysig; 
  if(sig_positive){
    mysig = sig;
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
  }else{
    mysig = OPP_DIR(sig);
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
  }
  point_c = (new_mem_idx >> 1);
  if (mu_positive){
    bc_link_nbr_idx = point_c;	
    reconstructSign(&bc_link_sign, mymu, new_x);
  }
  // So far, we have just computed ad_link_nbr_idx and 
  // bc_link_nbr_idx

  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];
  if(sig_positive){
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
  }else{
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
  }
  point_b = (new_mem_idx >> 1); 

  if (!mu_positive){
    bc_link_nbr_idx = point_b;
    reconstructSign(&bc_link_sign, mymu, new_x);
  }   

  if(sig_positive){
    ab_link_nbr_idx = sid;
    reconstructSign(&ab_link_sign, mysig, x);	
  }else{	
    ab_link_nbr_idx = point_b;
    reconstructSign(&ab_link_sign, mysig, new_x);
  }
  // now we have ab_link_nbr_idx


  // load the link variable connecting a and b 
  // Store in link_W 
  if(sig_positive){
    FF_LOAD_ARRAY(linkEven, mysig, ab_link_nbr_idx, LINK_W);	
  }else{
    FF_LOAD_ARRAY(linkOdd, mysig, ab_link_nbr_idx, LINK_W);	
  }
  RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, link_W);

  // load the link variable connecting b and c 
  // Store in link_X
  if(mu_positive){
    FF_LOAD_ARRAY(linkEven, mymu, bc_link_nbr_idx, LINK_X);
  }else{ 
    FF_LOAD_ARRAY(linkOdd, mymu, bc_link_nbr_idx, LINK_X);	
  }
  RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, link_X);



  LOAD_MATRIX_18_SINGLE(tempEven, point_c, COLOR_MAT_Y);
  // I do not think that Q3 is needed!
  if(mu_positive){
    ADJ_MAT_MUL_MAT(link_X, color_mat_Y, color_mat_W);
  }else{
    MAT_MUL_MAT(link_X, color_mat_Y, color_mat_W);
  }


  // Why write to PmuOdd instead of PmuEven?
  // Well, PmuEven would require tempOdd 
  // i.e., an extra device-memory access
  WRITE_MATRIX_18_SINGLE(PmuOdd, point_b, COLOR_MAT_W);
  if(sig_positive){
    MAT_MUL_MAT(link_W, color_mat_W, color_mat_Y);
  }else{ 
    ADJ_MAT_MUL_MAT(link_W, color_mat_W, color_mat_Y);
  }
  WRITE_MATRIX_18_SINGLE(P3Even, sid, COLOR_MAT_Y);


  if(mu_positive){
    FF_LOAD_ARRAY(linkOdd, mymu, ad_link_nbr_idx, LINK_Y);
    RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_Y);
  }else{
    FF_LOAD_ARRAY(linkEven, mymu, ad_link_nbr_idx, LINK_X);
    RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_X);
    ADJ_MAT(link_X, link_Y);
  }


  // if threeStaple - additional factorisation here!
  if(QprevOdd == NULL){
    if(sig_positive){
      MAT_MUL_MAT(color_mat_W, link_Y, color_mat_Y);
    }
    ASSIGN_MAT(link_Y, color_mat_W); 
    WRITE_MATRIX_18_SINGLE(QmuEven, sid, COLOR_MAT_W);
  }else{ 
    LOAD_MATRIX_18_SINGLE(QprevOdd, point_d, COLOR_MAT_Y);   
    MAT_MUL_MAT(color_mat_Y, link_Y, color_mat_X);
    WRITE_MATRIX_18_SINGLE(QmuEven, sid, COLOR_MAT_X);
    if(sig_positive){
      MAT_MUL_MAT(color_mat_W, color_mat_X, color_mat_Y);
    }	
  }

   
  if(sig_positive){
   const float & mycoeff = -CoeffSign<sig_positive,oddBit>::result*coeff;

   LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
   SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, color_mat_Y, mycoeff, color_mat_Z);
   WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
  }

  return;
}



  static void 
compute_force_kernel(float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
    float2* momMatrixEven, float2* momMatrixOdd,
    int sig, dim3 gridDim, dim3 blockDim,
    float2* momEven, float2* momOdd)
{
  dim3 halfGridDim(gridDim.x/2, 1, 1);

  // Need to see if this is necessary in the lates version of quda
  cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
  cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd,  cudaSiteLink.bytes);

  do_compute_force_kernel<0><<<halfGridDim, blockDim>>>(linkEven, linkOdd,
      momMatrixEven, momMatrixOdd,
      sig, 
      momEven, momOdd);
  cudaUnbindTexture(siteLink0TexSingle_recon);
  cudaUnbindTexture(siteLink1TexSingle_recon);

  cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
  cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

  do_compute_force_kernel<1><<<halfGridDim, blockDim>>>(linkOdd, linkEven,
      momMatrixOdd, momMatrixEven,
      sig,
      momOdd, momEven);

  cudaUnbindTexture(siteLink0TexSingle_recon);
  cudaUnbindTexture(siteLink1TexSingle_recon);

}





  static void
middle_link_kernel(const float2 * const tempEven, const float2 * const tempOdd, 
    float2 * const PmuEven,   float2 * const PmuOdd,
    float2 * const P3Even,    float2 * const P3Odd,
    const float2 * const QprevEven, const float2 * const QprevOdd,
    float2 * const QmuEven,   float2 * const QmuOdd,
    int sig, int mu, float coeff,
    float4 * const linkEven, float4 * const linkOdd, FullGauge cudaSiteLink,
    float2 * const  momEven, float2 * const momOdd,
    dim3 gridDim, dim3 BlockDim,
    float2 * const momMatrixEven, float2 * const momMatrixOdd)
{
  dim3 halfGridDim(gridDim.x/2, 1,1);

  cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
  cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);

  if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
    do_middle_link_kernel<1,1,0><<<halfGridDim, BlockDim>>>( tempEven,
        PmuOdd,  P3Even,
        QprevOdd,
        QmuEven, 
        sig, mu, coeff,
        linkEven, linkOdd,
        momEven, 
        momMatrixEven);
    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);
    //opposite binding
    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

    do_middle_link_kernel<1,1,1><<<halfGridDim, BlockDim>>>( tempOdd, 
        PmuEven,  P3Odd,
        QprevEven,
        QmuOdd, 
        sig, mu, coeff,
        linkOdd, linkEven,
        momOdd, 
        momMatrixOdd);
  }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
    do_middle_link_kernel<1,0,0><<<halfGridDim, BlockDim>>>( tempEven,
        PmuOdd,  P3Even,
        QprevOdd,
        QmuEven,
        sig, mu, coeff,
        linkEven, linkOdd,
        momEven, 
        momMatrixEven);	
    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);

    //opposite binding
    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

    do_middle_link_kernel<1,0,1><<<halfGridDim, BlockDim>>>( tempOdd, 
        PmuEven,  P3Odd,
        QprevEven,
        QmuOdd,  
        sig, mu, coeff,
        linkOdd, linkEven,
        momOdd, 
        momMatrixOdd);

  }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
    do_middle_link_kernel<0,1,0><<<halfGridDim, BlockDim>>>( tempEven, 
        PmuOdd,  P3Even,
        QprevOdd,
        QmuEven, 
        sig, mu, coeff,
        linkEven, linkOdd,
        momEven, 
        momMatrixEven);	
    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);

    //opposite binding
    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

    do_middle_link_kernel<0,1,1><<<halfGridDim, BlockDim>>>( tempOdd,
        PmuEven,  P3Odd,
        QprevEven, 
        QmuOdd, 
        sig, mu, coeff,
        linkOdd, linkEven,
        momOdd, 
        momMatrixOdd);
  }else{
    do_middle_link_kernel<0,0,0><<<halfGridDim, BlockDim>>>( tempEven,
        PmuOdd, P3Even,
        QprevOdd,
        QmuEven, 
        sig, mu, coeff,
        linkEven, linkOdd,
        momEven, 
        momMatrixEven);		

    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);

    //opposite binding
    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

    do_middle_link_kernel<0,0,1><<<halfGridDim, BlockDim>>>( tempOdd, 
        PmuEven,  P3Odd,
        QprevEven,
        QmuOdd,  
        sig, mu, coeff,
        linkOdd, linkEven,
        momOdd, 
        momMatrixOdd);		
  }
  cudaUnbindTexture(siteLink0TexSingle_recon);
  cudaUnbindTexture(siteLink1TexSingle_recon);    
}


template<int sig_positive, int mu_positive, int oddBit>
  __global__ void
do_side_link_kernel(const float2 * const P3Even, 
    const float2* const TempxEven, const float2 * const TempxOdd,
    float2 * const shortPOdd,
    int sig, int mu, float coeff, float accumu_coeff,
    const float4 * const linkEven, const float4 * const linkOdd,
    float2 * const momEven, float2 * const momOdd,
    float2 * const momMatrixEven, float2 * const momMatrixOdd)
{

  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int ad_link_sign = 1;

  float4 LINK_W[5];
  float2 COLOR_MAT_W[9], COLOR_MAT_X[9], COLOR_MAT_Y[9], COLOR_MAT_Z[9];

 
  /*    
   * 	  compute the side link contribution to the momentum
   *

   sig
   A________B
   |      |   mu
   D |      |C

   A is the current point (sid)
   */

  float mycoeff;
  int point_d;
  int ad_link_nbr_idx;
  int mymu;
  int new_mem_idx;

  int new_x[4];
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(mu_positive){
    mymu=mu;
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mymu,X, new_mem_idx);
  }else{
    mymu = OPP_DIR(mu);
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mymu, X, new_mem_idx);
  }
  point_d = (new_mem_idx >> 1);


  if (mu_positive){
    ad_link_nbr_idx = point_d;
    reconstructSign(&ad_link_sign, mymu, new_x);
  }else{
    ad_link_nbr_idx = sid;
    reconstructSign(&ad_link_sign, mymu, x);	
  }


  LOAD_MATRIX_18_SINGLE(P3Even, sid, COLOR_MAT_Y);
  if(mu_positive){
    FF_LOAD_ARRAY(linkOdd, mymu, ad_link_nbr_idx, LINK_W);
  }else{
    FF_LOAD_ARRAY(linkEven, mymu, ad_link_nbr_idx, LINK_W);
  }

  RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_W);	


  // Should all be inside if (shortPOdd)
  if (shortPOdd){
    if (mu_positive){
      MAT_MUL_MAT(link_W, color_mat_Y, color_mat_W);
    }else{
      ADJ_MAT_MUL_MAT(link_W, color_mat_Y, color_mat_W);
    }
    LOAD_MATRIX_18_SINGLE(shortPOdd, point_d, COLOR_MAT_X);
    SCALAR_MULT_ADD_MATRIX(color_mat_X, color_mat_W, accumu_coeff, color_mat_X);
    WRITE_MATRIX_18_SINGLE(shortPOdd, point_d, COLOR_MAT_X);
  }


  mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;

  if (mu_positive){
    if(TempxOdd){
      LOAD_MATRIX_18_SINGLE(TempxOdd, point_d, COLOR_MAT_X);
      MAT_MUL_MAT(color_mat_Y, color_mat_X, color_mat_W);
    }else{
      ASSIGN_MAT(color_mat_Y, color_mat_W);
    }
   
    LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, COLOR_MAT_Z);
    SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, color_mat_W, mycoeff, color_mat_Z);
    WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, COLOR_MAT_Z);
  }else{

    if(TempxOdd){
      LOAD_MATRIX_18_SINGLE(TempxOdd, point_d, COLOR_MAT_X);
      ADJ_MAT(color_mat_X,color_mat_W);
      MAT_MUL_ADJ_MAT(color_mat_W, color_mat_Y, color_mat_X);
    }else{
      ADJ_MAT(color_mat_Y, color_mat_X);
    }
    
    LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
    SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, color_mat_X, mycoeff, color_mat_Z);
    WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
  }
  return;
}





static void
side_link_kernel(float2* P3Even, float2* P3Odd, 
		 float2* TempxEven, float2* TempxOdd,
		 float2* shortPEven,  float2* shortPOdd,
		 int sig, int mu, float coeff, float accumu_coeff,
		 float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
		 float2* momEven, float2* momOdd,
		 dim3 gridDim, dim3 blockDim,
		 float2* momMatrixEven, float2* momMatrixOdd)
{
    dim3 halfGridDim(gridDim.x/2,1,1);
    
    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);   

    if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	do_side_link_kernel<1,1,0><<<halfGridDim, blockDim>>>( P3Even, 
							       TempxEven,  TempxOdd,
							       shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven, linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

	do_side_link_kernel<1,1,1><<<halfGridDim, blockDim>>>( P3Odd, 
							       TempxOdd,  TempxEven,
							       shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);
	
    }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	do_side_link_kernel<1,0,0><<<halfGridDim, blockDim>>>( P3Even, 
							       TempxEven,  TempxOdd,
							       shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven,  linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);		
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

	do_side_link_kernel<1,0,1><<<halfGridDim, blockDim>>>( P3Odd, 
							       TempxOdd,  TempxEven,
							       shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);		

    }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	do_side_link_kernel<0,1,0><<<halfGridDim, blockDim>>>( P3Even,
							       TempxEven,  TempxOdd,
							       shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven,  linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

	do_side_link_kernel<0,1,1><<<halfGridDim, blockDim>>>( P3Odd,
							       TempxOdd,  TempxEven,
							       shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);
	
    }else{
	do_side_link_kernel<0,0,0><<<halfGridDim, blockDim>>>( P3Even,
							       TempxEven,  TempxOdd,
							       shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven, linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
	
	do_side_link_kernel<0,0,1><<<halfGridDim, blockDim>>>( P3Odd, 
							       TempxOdd,  TempxEven,
							       shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);
    }
    
    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);    

}

template<int sig_positive, int mu_positive, int oddBit>
__global__ void
do_all_link_kernel(const float2* tempEven, 
		float2* QprevOdd,
		float2* PmuEven, float2* PmuOdd,
		float2* P3Even, float2* P3Odd,
		float2* P3muEven, float2* P3muOdd,
		float2* shortPEven, float2* shortPOdd,
		int sig, int mu, 
		float coeff, float mcoeff, float accumu_coeff,
		float4* linkEven, float4* linkOdd,
		float2* momEven, float2* momOdd,
		float2* momMatrixEven, float2* momMatrixOdd)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;

    int x[4];

    int z1 = sid/X1h;
    int x1h = sid - z1*X1h;
    int z2 = z1/X2;
    x[1] = z1 - z2*X2;
    x[3] = z2/X3;
    x[2] = z2 - x[3]*X3;
    int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
    x[0] = 2*x1h + x1odd;
    int X = 2*sid + x1odd;
    
    int new_x[4];
    int ad_link_sign=1;
    int ab_link_sign=1;
    int bc_link_sign=1;   
    
    float4 LINK_W[5], LINK_X[5], LINK_Y[5], LINK_Z[5];
    float2 COLOR_MAT_W[9], COLOR_MAT_Y[9], COLOR_MAT_X[9], COLOR_MAT_Z[9];
 

    /*       sig
           A________B
	mu  |      |
	  D |      |C
	  
	  A is the current point (sid)
    */
    int point_b, point_c, point_d;
    int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
    int mymu;
    int new_mem_idx;
    new_x[0] = x[0];
    new_x[1] = x[1];
    new_x[2] = x[2];
    new_x[3] = x[3];

    if(mu_positive){
	mymu =mu;
	FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
    }else{
	mymu = OPP_DIR(mu);
	FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(OPP_DIR(mu), X, new_mem_idx);	
    }
    point_d = (new_mem_idx >> 1);

    if (mu_positive){
	ad_link_nbr_idx = point_d;
	reconstructSign(&ad_link_sign, mymu, new_x);
    }else{
	ad_link_nbr_idx = sid;
	reconstructSign(&ad_link_sign, mymu, x);	
    }
  
 
    int mysig; 
    if(sig_positive){
	mysig = sig;
	FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
    }else{
	mysig = OPP_DIR(sig);
	FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
    }
    point_c = (new_mem_idx >> 1);
    if (mu_positive){
	bc_link_nbr_idx = point_c;	
	reconstructSign(&bc_link_sign, mymu, new_x);
    }
    
    new_x[0] = x[0];
    new_x[1] = x[1];
    new_x[2] = x[2];
    new_x[3] = x[3];
    if(sig_positive){
	FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
    }else{
	FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
    }
    point_b = (new_mem_idx >> 1);
    if (!mu_positive){
	bc_link_nbr_idx = point_b;
	reconstructSign(&bc_link_sign, mymu, new_x);
    }      
    
    if(sig_positive){
	ab_link_nbr_idx = sid;
	reconstructSign(&ab_link_sign, mysig, x);	
    }else{	
	ab_link_nbr_idx = point_b;
	reconstructSign(&ab_link_sign, mysig, new_x);
    }

    LOAD_MATRIX_18_SINGLE(QprevOdd, point_d, COLOR_MAT_X);
    ASSIGN_MAT(color_mat_X, link_W);
  
    if (mu_positive){
	FF_LOAD_ARRAY(linkOdd, mymu, ad_link_nbr_idx, LINK_Y);
    }else{
	FF_LOAD_ARRAY(linkEven, mymu, ad_link_nbr_idx, LINK_Y);
    }
    RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_Y);

    if (mu_positive){
	MAT_MUL_MAT(link_W, link_Y, color_mat_W);
    }else{
	MAT_MUL_ADJ_MAT(link_W, link_Y, color_mat_W);
    }
    LOAD_MATRIX_18_SINGLE(tempEven, point_c, COLOR_MAT_Y);


    if (mu_positive){
	FF_LOAD_ARRAY(linkEven, mymu, bc_link_nbr_idx, LINK_W);
    }else{
	FF_LOAD_ARRAY(linkOdd, mymu, bc_link_nbr_idx, LINK_W);	
    }
    RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, link_W);


    if (mu_positive){    
	ADJ_MAT_MUL_MAT(link_W, color_mat_Y, link_X);
    }else{
	MAT_MUL_MAT(link_W, color_mat_Y, link_X);
    }
    // link_X now connects site b to the outer product! 
    // Done with LINK_W for the time being.	

    if (sig_positive){
	FF_LOAD_ARRAY(linkEven, mysig, ab_link_nbr_idx, LINK_W);
    }else{
	FF_LOAD_ARRAY(linkOdd, mysig, ab_link_nbr_idx, LINK_W);
    }
    RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, link_W);


   if (sig_positive){        
     MAT_MUL_MAT(link_W, link_X, color_mat_Y);
   }else{
     ADJ_MAT_MUL_MAT(link_W, link_X, color_mat_Y);
   }
    // color_mat_Y now connects site a to the outer product 
    // Force from the forward link in the staple


   const float & mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;
   if (sig_positive)
   {	
     MAT_MUL_MAT(link_X, color_mat_W, link_Z);
     ASSIGN_MAT(link_Z, color_mat_W);
     LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
     SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
     WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
   }

   // Note that mu is known at compile time 
   // Surely this code can be made more generic
   // QprevOdd = color_mat_X
   if (mu_positive)
   {
     MAT_MUL_MAT(color_mat_Y, color_mat_X, link_Z);
     LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, COLOR_MAT_Z);
     SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
     WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, COLOR_MAT_Z);
     MAT_MUL_MAT(link_Y, color_mat_Y, color_mat_W);	
   }else
   {
     ADJ_MAT_MUL_ADJ_MAT(color_mat_X, color_mat_Y, link_Z);	
     LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
     SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
     WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
     ADJ_MAT_MUL_MAT(link_Y, color_mat_Y, color_mat_W);	
   }

   LOAD_MATRIX_18_SINGLE(shortPOdd, point_d, COLOR_MAT_Y);
   SCALAR_MULT_ADD_MATRIX(color_mat_Y, color_mat_W, accumu_coeff, color_mat_Y);
   WRITE_MATRIX_18_SINGLE(shortPOdd, point_d, COLOR_MAT_Y);

   return;
}



static void
all_link_kernel(const float2* link_ZxEven, const float2* link_ZxOdd,
		float2* QprevEven, float2* QprevOdd, 
		float2* PmuEven, float2* PmuOdd,
		float2* P3Even, float2* P3Odd,
		float2* P3muEven, float2* P3muOdd,
		float2* shortPEven, float2* shortPOdd,
		int sig, int mu,
		float coeff, float mcoeff, float accumu_coeff,
		float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
		float2* momEven, float2* momOdd,
		dim3 gridDim, dim3 blockDim,
		float2* momMatrixEven, float2* momMatrixOdd)
		   
{
    dim3 halfGridDim(gridDim.x/2, 1,1);

    cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
    
    if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){		
	do_all_link_kernel<1,1,0><<<halfGridDim, blockDim>>>( link_ZxEven,  
							      QprevOdd, 
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu,
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd,
							      momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
	do_all_link_kernel<1,1,1><<<halfGridDim, blockDim>>>( link_ZxOdd,  
							      QprevEven,
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu,
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);	

	
    }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){

	do_all_link_kernel<1,0,0><<<halfGridDim, blockDim>>>( link_ZxEven,   
							      QprevOdd,
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, 
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd,
							      momMatrixEven, momMatrixOdd);	
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

	do_all_link_kernel<1,0,1><<<halfGridDim, blockDim>>>( link_ZxOdd,  
							      QprevEven, 
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, 
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);	
	
    }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	do_all_link_kernel<0,1,0><<<halfGridDim, blockDim>>>( link_ZxEven,  
							      QprevOdd, 
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, 
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd, 
							      momMatrixEven, momMatrixOdd);	
	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

	
	do_all_link_kernel<0,1,1><<<halfGridDim, blockDim>>>( link_ZxOdd,  
							      QprevEven, 
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, 
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);		
    }else{
	do_all_link_kernel<0,0,0><<<halfGridDim, blockDim>>>( link_ZxEven, 
							      QprevOdd, 
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, 
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd,
							      momMatrixEven, momMatrixOdd);	

	cudaUnbindTexture(siteLink0TexSingle_recon);
	cudaUnbindTexture(siteLink1TexSingle_recon);

	//opposite binding
	cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

	do_all_link_kernel<0,0,1><<<halfGridDim, blockDim>>>( link_ZxOdd,  
							      QprevEven, 
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, 
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);	
    }

    cudaUnbindTexture(siteLink0TexSingle_recon);
    cudaUnbindTexture(siteLink1TexSingle_recon);
}

/*
__global__ void
one_and_naik_terms_kernel(float2* TempxEven, float2* TempxOdd,
			  float2* PmuEven,   float2* PmuOdd, 
			  float2* PnumuEven, float2* PnumuOdd,
			  int mu, float OneLink, float Naik, float mNaik,
			  float4* linkEven, float4* linkOdd,
			  float2* momEven, float2* momOdd)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    int oddBit = 0;
    float2* myTempx = TempxEven;
    float2* myPmu = PmuEven;
    float2* myPnumu = PnumuEven;
    float2* myMom = momEven;
    float4* myLink = linkEven;    
    float2* otherTempx = TempxOdd;
    float2* otherPnumu = PnumuOdd;
    float4* otherLink = linkOdd;
    
    float2 HWA0, HWA1, HWA2, HWA3, HWA4, HWA5;
    float2 HWB0, HWB1, HWB2, HWB3, HWB4, HWB5;
    float2 HWC0, HWC1, HWC2, HWC3, HWC4, HWC5;
    float2 HWD0, HWD1, HWD2, HWD3, HWD4, HWD5;
    float4 LINK_W0, LINK_W1, LINK_W2, LINK_W3, LINK_W4;
    float4 LINK_X0, LINK_X1, LINK_X2, LINK_X3, LINK_X4;
    float2 AH0, AH1, AH2, AH3, AH4;    
    
    if (sid >= Vh){
        oddBit =1;
        sid -= Vh;
	
	myTempx = TempxOdd;
	myPmu = PmuOdd;
	myPnumu = PnumuOdd;
	myMom = momOdd;
	myLink = linkOdd;  	
	otherTempx = TempxEven;
	otherPnumu = PnumuEven;
	otherLink = linkEven;
    }
    
    int z1 = sid/X1h;
    int x1h = sid - z1*X1h;
    int z2 = z1/X2;
    int x2 = z1 - z2*X2;
    int x4 = z2/X3;
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    //int X = 2*sid + x1odd;
    
    int dx[4];
    int new_x[0], new_x[1], new_x[2], new_x[3], new_idx;
    int sign=1;
    
    if (GOES_BACKWARDS(mu)){
	//The one link
	LOAD_HW(myPmu, sid, HWA);
	LOAD_HW(myTempx, sid, HWB);
	ADD_FORCE_TO_MOM(hwa, hwb, myMom, sid, OPP_DIR(mu), OneLink, oddBit);
	
	//Naik term
	dx[3]=dx[2]=dx[1]=dx[0]=0;
	dx[OPP_DIR(mu)] = -1;
	new_x[0] = (x1 + dx[0] + X1)%X1;
	new_x[1] = (x2 + dx[1] + X2)%X2;
	new_x[2] = (x3 + dx[2] + X3)%X3;
	new_x[3] = (x4 + dx[3] + X4)%X4;	
	new_idx = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;
	LOAD_HW(otherTempx, new_idx, HWA);
	LOAD_MATRIX(otherLink, OPP_DIR(mu), new_idx, LINK_W);
	reconstructSign(sign, OPP_DIR(mu), new_x[0],new_x[1],new_x[2],new_x[3]);
	RECONSTRUCT_LINK_12(OPP_DIR(mu), new_idx, sign, link_W);		
	ADJ_MAT_MUL_HW(link_W, hwa, hwc); //Popmu
	
	LOAD_HW(myPnumu, sid, HWD);
	ADD_FORCE_TO_MOM(hwd, hwc, myMom, sid, OPP_DIR(mu), mNaik, oddBit);
	
	dx[3]=dx[2]=dx[1]=dx[0]=0;
	dx[OPP_DIR(mu)] = 1;
	new_x[0] = (x1 + dx[0] + X1)%X1;
	new_x[1] = (x2 + dx[1] + X2)%X2;
	new_x[2] = (x3 + dx[2] + X3)%X3;
	new_x[3] = (x4 + dx[3] + X4)%X4;	
	new_idx = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;
	LOAD_HW(otherPnumu, new_idx, HWA);
	LOAD_MATRIX(myLink, OPP_DIR(mu), sid, LINK_W);
	reconstructSign(sign, OPP_DIR(mu), x1, x2, x3, x4);
	RECONSTRUCT_LINK_12(OPP_DIR(mu), sid, sign, link_W);	
	MAT_MUL_HW(link_W, hwa, hwc);
	ADD_FORCE_TO_MOM(hwc, hwb, myMom, sid, OPP_DIR(mu), Naik, oddBit);	
    }else{
	dx[3]=dx[2]=dx[1]=dx[0]=0;
	dx[mu] = 1;
	new_x[0] = (x1 + dx[0] + X1)%X1;
	new_x[1] = (x2 + dx[1] + X2)%X2;
	new_x[2] = (x3 + dx[2] + X3)%X3;
	new_x[3] = (x4 + dx[3] + X4)%X4;	
	new_idx = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;
	LOAD_HW(otherTempx, new_idx, HWA);
	LOAD_MATRIX(myLink, mu, sid, LINK_W);
	reconstructSign(sign, mu, x1, x2, x3, x4);
	RECONSTRUCT_LINK_12(mu, sid, sign, link_W);
	MAT_MUL_HW(link_W, hwa, hwb);
	
	LOAD_HW(myPnumu, sid, HWC);
	ADD_FORCE_TO_MOM(hwb, hwc, myMom, sid, mu, Naik, oddBit);
	

    }
}
*/


#define Pmu 	  tempmat[0]
#define P3        tempmat[1]
#define P5	  tempmat[2]
#define Pnumu     tempmat[3]
#define P3mu	  tempmat[3]
#define P5nu	  tempmat[3]
#define P7 	  tempmat[3]
#define Prhonumu  tempmat[3]
#define P7rho     tempmat[3]


// Here, we have a problem. 
// If we use float2 to store the Ps and float2 
// for the Qs. We can't use the same link_Zoraries 
// here. 
// I wonder which is better? 
// To use float2 for both Ps and Qs and 
// and use the same link_Zorary matrices 
// or use float2 for the Ps and float4 for the 
// Qs and use separate sets of matrices?
// Ultimately, I will use float4 for the Q matrices 
// for the first level of smearing and float2 
// for the Q matrices for the second level of smearing. 
// To begin with, use float4 for everything.
// Note, I will have to go back and undo the float4s above.


// if first level of smearing
 #define Qmu      tempCmat[0]
 #define Qnumu    tempCmat[1]
 #define Qrhonumu tempCmat[2] 
 #define Q5       tempCmat[2]


// tempCmat should be a full compressed matrix
// FullCompMat

// if !first level of smearing
//#define Qmu	  tempmat[7]
//#define Qnumu	  tempmat[8]
//#define Qrhonumu  tempmat[2] // same as Prhonumu

// Need to define new types 
// FullMat 
// FullCompMat

template<typename Real>
static void
do_hisq_force_cuda(Real eps, Real weight1, Real weight2,  Real* act_path_coeff, FullOprod cudaOprod, // need to change this code
		      FullGauge cudaSiteLink, FullMom cudaMom, FullGauge cudaMomMatrix, FullMatrix tempmat[7], FullMatrix tempCmat[4], QudaGaugeParam* param)
{
    
    int mu, nu, rho, sig;
    float coeff;
    
    float OneLink, Lepage, Naik, FiveSt, ThreeSt, SevenSt;
    float mLepage, mNaik, mFiveSt, mThreeSt, mSevenSt;
    
    Real ferm_epsilon;
    ferm_epsilon = 2.0*weight1*eps;
    OneLink = act_path_coeff[0]*ferm_epsilon ;
    Naik    = act_path_coeff[1]*ferm_epsilon ; mNaik    = -Naik;
    ThreeSt = act_path_coeff[2]*ferm_epsilon ; mThreeSt = -ThreeSt;
    FiveSt  = act_path_coeff[3]*ferm_epsilon ; mFiveSt  = -FiveSt;
    SevenSt = act_path_coeff[4]*ferm_epsilon ; mSevenSt = -SevenSt;
    Lepage  = act_path_coeff[5]*ferm_epsilon ; mLepage  = -Lepage;
    
    int DirectLinks[8] ;    
    
    for(mu=0;mu<8;mu++){
        DirectLinks[mu] = 0 ;
    }
        
    int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim(volume/blockDim.x, 1, 1);
   
    int null = -1;
   
    for(sig=0; sig < 8; sig++){
        for(mu = 0; mu < 8; mu++){
            if ( (mu == sig) || (mu == OPP_DIR(sig))){
                continue;
            }
	    //3-link
	    //Kernel A: middle link
	   
	    middle_link_kernel( (float2*)cudaOprod.even.data[OPP_DIR(sig)], (float2*)cudaOprod.odd.data[OPP_DIR(sig)],
				(float2*)Pmu.even.data, (float2*)Pmu.odd.data,
				(float2*)P3.even.data, (float2*)P3.odd.data,
				(float2*)NULL,         (float2*)NULL,
				(float2*)Qmu.even.data, (float2*)Qmu.odd.data,
				sig, mu, mThreeSt,
				(float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
				(float2*)cudaMom.even, (float2*)cudaMom.odd, 
				gridDim, blockDim,
			        (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); // I have added true and false to indicate 
							                                  // whether I am on a three staple
                                                                                          // Actually, I just have to check if the pointer   
                                                                                          // to the previous path is a NULL pointer!     



	
	    checkCudaError();

            for(nu=0; nu < 8; nu++){
                if (nu == sig || nu == OPP_DIR(sig)
                    || nu == mu || nu == OPP_DIR(mu)){
                    continue;
                }

		//5-link: middle link
		//Kernel B
		middle_link_kernel( (float2*)Pmu.even.data, (float2*)Pmu.odd.data,
				    (float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
				    (float2*)P5.even.data, (float2*)P5.odd.data,
				    (float2*)Qmu.even.data, (float2*)Qmu.odd.data, // input Q matrix
				    (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				    sig, nu, FiveSt,
				    (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
				    (float2*)cudaMom.even, (float2*)cudaMom.odd,
				    gridDim, blockDim,
				    (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); // no longer on a threeStaple => have to read in Qprev

		checkCudaError();

                for(rho =0; rho < 8; rho++){
                    if (rho == sig || rho == OPP_DIR(sig)
                        || rho == mu || rho == OPP_DIR(mu)
                        || rho == nu || rho == OPP_DIR(nu)){
                        continue;
                    }
		    //7-link: middle link and side link
		    //kernel C
		    if(FiveSt != 0)coeff = SevenSt/FiveSt ; else coeff = 0;
		    all_link_kernel((float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
				    (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				    (float2*)Prhonumu.even.data, (float2*)Prhonumu.odd.data,
				    (float2*)P7.even.data, (float2*)P7.odd.data,
				    (float2*)P7rho.even.data, (float2*)P7rho.odd.data,
				    (float2*)P5.even.data, (float2*)P5.odd.data,
				    sig, rho, SevenSt, mSevenSt, coeff,
				    (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
				    (float2*)cudaMom.even, (float2*)cudaMom.odd,
				    gridDim, blockDim,
				    (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);	
		    checkCudaError();

		}//rho  		
                // P7, P7rho, P7rhonumu are free to be used again


		//5-link: side link
		//kernel B2
		if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
		side_link_kernel((float2*)P5.even.data, (float2*)P5.odd.data,
		//		 (float2*)P5nu.even.data, (float2*)P5nu.odd.data, // output
				 (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
	         //	        (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				 (float2*)P3.even.data, (float2*)P3.odd.data,
				 sig, nu, mFiveSt, coeff,
				 (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
				 (float2*)cudaMom.even, (float2*)cudaMom.odd,
				 gridDim, blockDim,
				 (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
		checkCudaError();



	    } //nu 
            // P5nu, Pnumu are free to be used again

	    //lepage
	    //Kernel A2
	    middle_link_kernel( (float2*)Pmu.even.data, (float2*)Pmu.odd.data,
				(float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
				(float2*)P5.even.data, (float2*)P5.odd.data,
				(float2*)Qmu.even.data, (float2*)Qmu.odd.data, // input Q matrix
				(float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				sig, mu, Lepage,
				(float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
				(float2*)cudaMom.even, (float2*)cudaMom.odd,
				gridDim, blockDim, 
				(float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); // not on a threeStaple => have to read in Qprev   
	    checkCudaError();		
	    
	    if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;
	    
	    side_link_kernel((float2*)P5.even.data, (float2*)P5.odd.data,
	//		     (float2*)P5nu.even.data, (float2*)P5nu.odd.data,
			     (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
	//		     (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
			     (float2*)P3.even.data, (float2*)P3.odd.data,
			     sig, mu, mLepage ,coeff,
			     (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
			     (float2*)cudaMom.even, (float2*)cudaMom.odd,
			     gridDim, blockDim,
			     (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
	    checkCudaError();		


	    //3-link side link
	    coeff=0.;

	    side_link_kernel((float2*)P3.even.data, (float2*)P3.odd.data,
	//		     (float2*)P3mu.even.data, (float2*)P3mu.odd.data,
			     (float2*)NULL, (float2*)NULL,
	//		     (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
			     (float2*)NULL, (float2*)NULL,
			     sig, mu, ThreeSt, coeff,
			     (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
			     (float2*)cudaMom.even, (float2*)cudaMom.odd,
			     gridDim, blockDim,
			     (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
	    checkCudaError();			    




//	    //1-link and naik term	    
//	    if (!DirectLinks[mu]){
//		DirectLinks[mu]=1;
//		//kernel Z	    
//		one_and_naik_terms_kernel<<<gridDim, blockDim>>>((float2*)cudaHw.even.data, (float2*)cudaHw.odd.data,
//								 (float2*)Pmu.even.data, (float2*)Pmu.odd.data,
//								 (float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
//								 mu, OneLink.x, Naik.x, mNaik.x, 
//								 (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd,
//								 (float2*)cudaMom.even, (float2*)cudaMom.odd);
//		checkCudaError();		
//	    }
	}//mu

    }//sig

    for(sig=0; sig<8; sig++){
      if(GOES_FORWARDS(sig)){
        compute_force_kernel( (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
                              (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd,
                              sig, gridDim, blockDim,
                              (float2*)cudaMom.even, (float2*)cudaMom.odd);
      } // Only compute the force term if it goes forwards
    } // sig
    
    
}

#undef Pmu
#undef Pnumu
#undef Prhonumu
#undef P3
#undef P3mu
#undef P5
#undef P5nu
#undef P7
#undef P7rho

#undef Qmu
#undef Qnumu
#undef Qrhonumu


void
hisq_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
		   FullOprod cudaOprod, FullGauge cudaSiteLink, FullMom cudaMom, FullGauge cudaMomMatrix, QudaGaugeParam* param)
{

    FullMatrix tempmat[4];
    for(int i=0; i<4; i++){
	tempmat[i]  = createMatQuda(param->X, param->cuda_prec);
    }

    FullMatrix tempCompmat[3];
    for(int i=0; i<3; i++){
 	tempCompmat[i] = createMatQuda(param->X, param->cuda_prec);
    }	


    if (param->cuda_prec == QUDA_DOUBLE_PRECISION){
    }else{	
	do_hisq_force_cuda( (float)eps, (float)weight1, (float)weight2, (float*)act_path_coeff,
			     cudaOprod,
			     cudaSiteLink, cudaMom, cudaMomMatrix, tempmat, tempCompmat, param);
    }
    
    for(int i=0; i<7; i++){
      freeMatQuda(tempmat[i]);
    }

    for(int i=0; i<4; i++){
      freeMatQuda(tempCompmat[i]);
    }
    return; 
}

} // namespace fermion_force
} // namespace hisq
