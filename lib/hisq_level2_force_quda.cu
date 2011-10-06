#include <read_gauge.h>
#include <gauge_quda.h>

#include "hisq_force_quda.h"
#include "force_common.h"
#include "hw_quda.h"
#include "hisq_force_macros.h"


#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_SINGLE
#define LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)

//#define FF_SITE_MATRIX_LOAD_TEX 1

#if (FF_SITE_MATRIX_LOAD_TEX == 1)
#define linkEvenTex siteLink0TexSingle_recon
#define linkOddTex siteLink1TexSingle_recon
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(src##Tex, dir, idx, var)
#define FF_LOAD_ARRAY(src, dir, idx, var) LOAD_ARRAY_12_SINGLE_TEX(src##Tex, dir, idx, var)    
#else
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)
#define FF_LOAD_ARRAY(src, dir, idx, var) LOAD_ARRAY_12_SINGLE(src, dir, idx, var)    
#endif


template<class T>
inline __device__
void loadMatrixFromField(T* const mat, int dir, int idx, const T* const field)
{
  mat[0] = field[idx + dir*Vhx3];
  mat[1] = field[idx + dir*Vhx3 + Vh];
  mat[2] = field[idx + dir*Vhx3 + Vhx2];
  mat[3] = field[idx + dir*Vhx3 + Vhx3];
  mat[4] = field[idx + dir*Vhx3 + Vhx4];
  mat[5] = field[idx + dir*Vhx3 + Vhx5];
  mat[6] = field[idx + dir*Vhx3 + Vhx6];
  mat[7] = field[idx + dir*Vhx3 + Vhx7];
  mat[8] = field[idx + dir*Vhx3 + Vhx8];

  return;
}


inline __device__
void loadMatrixFromField(float4* const mat, int dir, int idx, const float4* const field)
{
  mat[0] = field[idx + dir*Vhx3];
  mat[1] = field[idx + dir*Vhx3 + Vh];
  mat[2] = field[idx + dir*Vhx3 + Vhx2];
  return;
}



template<class T>
inline __device__
void loadMatrixFromField(T* const mat, int idx, const T* const field)
{
  mat[0] = field[idx];
  mat[1] = field[idx + Vh];
  mat[2] = field[idx + Vhx2];
  mat[3] = field[idx + Vhx3];
  mat[4] = field[idx + Vhx4];
  mat[5] = field[idx + Vhx5];
  mat[6] = field[idx + Vhx6];
  mat[7] = field[idx + Vhx7];
  mat[8] = field[idx + Vhx8];
 
  return;
}



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



namespace hisq {
  namespace fermion_force {



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

    template<class RealX>
      struct ArrayLength
      {
        static const int result=9;
        static const bool compressed=false;
      };

    template<>
      struct ArrayLength<float4>
      {
        static const int result=5;
        static const bool compressed=true;
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




    template<class RealA, class RealB, int oddBit>
      __global__ void 
      do_compute_force_kernel(const RealB* const linkEven, 
          const RealB* const linkOdd,
          const RealA* const momMatrixEven,      
          const RealA* const momMatrixOdd,
          int sig,
          RealA* const momEven, 
          RealA* const momOdd)
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

        RealB LINK_W[ArrayLength<RealB>::result];
        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        RealA COLOR_MAT_X[ArrayLength<RealA>::result];


        loadMatrixFromField(LINK_W, sig, sid, linkEven);
        reconstructSign(&link_sign, sig, x);	
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(sig, sid, link_sign, link_W);
        }

        LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_X);
        MAT_MUL_MAT(link_W, color_mat_X, color_mat_W);

        SIMPLE_MAT_FORCE_TO_MOM(color_mat_W, momEven, sid, sig, link_W);

        return;
      }

    template<class RealA, int oddBit>
      __global__ void 
      do_one_and_naik_terms_kernel(const RealA* const oprodEven, 
          int sig, float coeff, float naik_coeff,
          RealA* const momMatrixEven)
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
        new_x[0] = x[0];
        new_x[1] = x[1];
        new_x[2] = x[2];
        new_x[3] = x[3];

        int new_mem_idx;
        int point_b;

        if(GOES_FORWARDS(sig)){
          FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
        }else{
          FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
        }
        point_b = (new_mem_idx >> 1); 
        const int & point_a = sid;

        // May need to change this to properly take account of the Naik term!
        const float &  mycoeff = -CoeffSign<1,oddBit>::result*(coeff + naik_coeff);

        RealA COLOR_MAT_W[9], COLOR_MAT_Y[9];

        if(GOES_FORWARDS(sig)){
          LOAD_MATRIX_18_SINGLE(oprodEven, point_a, COLOR_MAT_W);
          ADJ_MAT(color_mat_W, color_mat_Y);
          LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, point_a, COLOR_MAT_W);
          SCALAR_MULT_ADD_SU3_MATRIX(color_mat_W, color_mat_Y, mycoeff, color_mat_W);
          WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, point_a, COLOR_MAT_W);
        }
        return;
      }


    template<class RealA>
      static void
      one_and_naik_terms(const RealA* const oprodEven, 
          const RealA* const oprodOdd,
          int sig, float coeff, float naik_coeff,
          dim3 gridDim, dim3 blockDim,
          RealA* const MomMatrixEven,   
          RealA* const  MomMatrixOdd)
      {

        dim3 halfGridDim(gridDim.x/2,1,1);

        if(GOES_FORWARDS(sig)){

          do_one_and_naik_terms_kernel<RealA,0><<<halfGridDim,blockDim>>>(oprodEven,
              sig, coeff, naik_coeff,
              MomMatrixEven);

          do_one_and_naik_terms_kernel<RealA, 1><<<halfGridDim,blockDim>>>(oprodOdd,
              sig, coeff, naik_coeff,
              MomMatrixOdd);

        } // GOES_FORWARDS(sig)

        return;
      }



    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit> 
      __global__ void
      do_middle_link_kernel(
          const RealA* const tempEven, 
          const RealA* const tempOdd,
          RealA* const PmuOdd, 
          RealA* const P3Even,
          RealA* const QprevOdd, 		
          RealA* const QmuEven, 
          int sig, int mu, float coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          RealA* const momMatrixEven 
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

        RealB LINK_W[ArrayLength<RealB>::result];
        RealB LINK_X[ArrayLength<RealB>::result];
        RealB LINK_Y[ArrayLength<RealB>::result];


        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result];
        RealA COLOR_MAT_X[ArrayLength<RealA>::result];
        RealA COLOR_MAT_Z[ArrayLength<RealA>::result];


        //        A________B
        //    mu   |      |
        // 	  D|      |C
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
          mymu = mu;
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
        // now we have ab_link_nbr_idx


        // load the link variable connecting a and b 
        // Store in link_W 
        if(sig_positive){
          loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkEven);
        }else{
          loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkOdd);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, link_W);
        }

        // load the link variable connecting b and c 
        // Store in link_X
        if(mu_positive){
          loadMatrixFromField(LINK_X, mymu, bc_link_nbr_idx, linkEven);
        }else{ 
          loadMatrixFromField(LINK_X, mymu, bc_link_nbr_idx, linkOdd);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, link_X);
        }


        if(QprevOdd == NULL && sig_positive){
          LOAD_MATRIX_18_SINGLE(tempOdd, point_d, COLOR_MAT_Z); 
          ADJ_MAT(color_mat_Z, color_mat_Y);
        }else{
          LOAD_MATRIX_18_SINGLE(tempEven, point_c, COLOR_MAT_Y);
        }


        if(mu_positive){
          ADJ_MAT_MUL_MAT(link_X, color_mat_Y, color_mat_W);
        }else{
          MAT_MUL_MAT(link_X, color_mat_Y, color_mat_W);
        }
        WRITE_MATRIX_18_SINGLE(PmuOdd, point_b, COLOR_MAT_W);

        if(sig_positive){
          MAT_MUL_MAT(link_W, color_mat_W, color_mat_Y);
        }else{ 
          ADJ_MAT_MUL_MAT(link_W, color_mat_W, color_mat_Y);
        }
        WRITE_MATRIX_18_SINGLE(P3Even, sid, COLOR_MAT_Y);


        if(mu_positive){
          loadMatrixFromField(LINK_Y, mymu, ad_link_nbr_idx, linkOdd);
          if(ArrayLength<RealB>::compressed){
            RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_Y);
          }
        }else{
          loadMatrixFromField(LINK_X, mymu, ad_link_nbr_idx, linkEven);
          if(ArrayLength<RealB>::compressed){
            RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_X);
          }
          ADJ_MAT(link_X, link_Y);
        }


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


    template<class RealA, class RealB>
      static void 
      compute_force_kernel(const RealB* const linkEven, 
          const RealB* const linkOdd, 
          FullGauge cudaSiteLink,
          const RealA* const momMatrixEven, 
          const RealA* const momMatrixOdd,
          int sig, dim3 gridDim, dim3 blockDim,
          RealA* const momEven, 
          RealA* const momOdd)
      {
        dim3 halfGridDim(gridDim.x/2, 1, 1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd,  cudaSiteLink.bytes);

        do_compute_force_kernel<RealA,RealB,0><<<halfGridDim, blockDim>>>(linkEven, linkOdd,
            momMatrixEven, momMatrixOdd,
            sig, 
            momEven, momOdd);
        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

        do_compute_force_kernel<RealA,RealB,1><<<halfGridDim, blockDim>>>(linkOdd, linkEven,
            momMatrixOdd, momMatrixEven,
            sig,
            momOdd, momEven);

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);

        return;
      }



    template<class RealA, class RealB>
      static void
      middle_link_kernel(const RealA* const tempEven, 
          const RealA* const tempOdd, 
          RealA* const PmuEven,   
          RealA* const PmuOdd,
          RealA* const P3Even,    
          RealA* const P3Odd,
          RealA* const QprevEven, 
          RealA* const QprevOdd,
          RealA* const QmuEven,   
          RealA* const QmuOdd,
          int sig, int mu, float coeff,
          RealB* const linkEven, 
          RealB* const linkOdd, 
          FullGauge cudaSiteLink,
          dim3 gridDim, dim3 BlockDim,
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd)
      {
        dim3 halfGridDim(gridDim.x/2, 1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
          do_middle_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, BlockDim>>>( tempEven, tempOdd,
              PmuOdd,  P3Even,
              QprevOdd,
              QmuEven, 
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);
          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_middle_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, BlockDim>>>( tempOdd, tempEven,
              PmuEven,  P3Odd,
              QprevEven,
              QmuOdd, 
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);
        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_middle_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, BlockDim>>>( tempEven, tempOdd,
              PmuOdd,  P3Even,
              QprevOdd,
              QmuEven,
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);	
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_middle_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, BlockDim>>>( tempOdd, tempEven,
              PmuEven,  P3Odd,
              QprevEven,
              QmuOdd,  
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
          do_middle_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, BlockDim>>>( tempEven, tempOdd,
              PmuOdd,  P3Even,
              QprevOdd,
              QmuEven, 
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);	
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_middle_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, BlockDim>>>( tempOdd, tempEven,
              PmuEven,  P3Odd,
              QprevEven, 
              QmuOdd, 
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);
        }else{
          do_middle_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, BlockDim>>>( tempEven, tempOdd,
              PmuOdd, P3Even,
              QprevOdd,
              QmuEven, 
              sig, mu, coeff,
              linkEven, linkOdd,
              momMatrixEven);		

          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_middle_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, BlockDim>>>( tempOdd, tempEven,
              PmuEven,  P3Odd,
              QprevEven,
              QmuOdd,  
              sig, mu, coeff,
              linkOdd, linkEven,
              momMatrixOdd);		
        }
        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }



    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_side_link_kernel(const RealA* const P3Even, 
          const RealA* const TempxEven, 
          const RealA* const TempxOdd,
          RealA* const shortPOdd,
          int sig, int mu, float coeff, float accumu_coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          RealA* const momEven, 
          RealA* const momOdd,
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd)
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

        RealB LINK_W[ArrayLength<RealB>::result];

        RealA COLOR_MAT_W[ArrayLength<RealA>::result];
        RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Z[ArrayLength<RealA>::result];

//      compute the side link contribution to the momentum
//
//             sig
//          A________B
//           |      |   mu
//         D |      |C
//
//      A is the current point (sid)

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
          loadMatrixFromField(LINK_W, mymu, ad_link_nbr_idx, linkOdd);
        }else{
          loadMatrixFromField(LINK_W, mymu, ad_link_nbr_idx, linkEven);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_W);	
        }


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




    template<class RealA, class RealB>
      static void
      side_link_kernel(RealA* P3Even, 
          RealA* P3Odd, 
          RealA* TempxEven, 
          RealA* TempxOdd,
          RealA* shortPEven,  
          RealA* shortPOdd,
          int sig, int mu, float coeff, float accumu_coeff,
          RealB* linkEven, 
          RealB* linkOdd, 
          FullGauge cudaSiteLink,
          RealA* momEven, 
          RealA* momOdd,
          dim3 gridDim, dim3 blockDim,
          RealA* momMatrixEven, 
          RealA* momMatrixOdd)
      {
        dim3 halfGridDim(gridDim.x/2,1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);   

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( P3Even, 
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

          do_side_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              sig, mu, coeff, accumu_coeff,
              linkOdd, linkEven,
              momOdd, momEven,
              momMatrixOdd, momMatrixEven);

        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( P3Even, 
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

          do_side_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              sig, mu, coeff, accumu_coeff,
              linkOdd, linkEven,
              momOdd, momEven,
              momMatrixOdd, momMatrixEven);		

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( P3Even,
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

          do_side_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( P3Odd,
              TempxOdd,  TempxEven,
              shortPEven,
              sig, mu, coeff, accumu_coeff,
              linkOdd, linkEven,
              momOdd, momEven,
              momMatrixOdd, momMatrixEven);

        }else{
          do_side_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( P3Even,
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

          do_side_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              sig, mu, coeff, accumu_coeff,
              linkOdd, linkEven,
              momOdd, momEven,
              momMatrixOdd, momMatrixEven);
        }

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }


    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_all_link_kernel(const RealA* const tempEven, 
          RealA* const QprevOdd,
          RealA* const PmuEven, 
          RealA* const PmuOdd,
          RealA* const P3Even, 
          RealA* const P3Odd,
          RealA* const P3muEven, 
          RealA* const P3muOdd,
          RealA* const shortPEven, 
          RealA* const shortPOdd,
          int sig, int mu, 
          float coeff, float mcoeff, float accumu_coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          RealA* const momEven, 
          RealA* const momOdd,
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd)
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

        RealB LINK_W[ArrayLength<RealB>::result];
        RealB LINK_X[ArrayLength<RealB>::result];
        RealB LINK_Y[ArrayLength<RealB>::result];
        RealB LINK_Z[ArrayLength<RealB>::result];

        RealA COLOR_MAT_W[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 
        RealA COLOR_MAT_Z[ArrayLength<RealA>::result];


        //            sig
        //         A________B
        //      mu  |      |
        //        D |      |C
        //
        //   A is the current point (sid)
        //

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
          loadMatrixFromField(LINK_Y, mymu, ad_link_nbr_idx, linkOdd);
        }else{
          loadMatrixFromField(LINK_Y, mymu, ad_link_nbr_idx, linkEven);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, link_Y);
        }

        if (mu_positive){
          MAT_MUL_MAT(link_W, link_Y, color_mat_W);
        }else{
          MAT_MUL_ADJ_MAT(link_W, link_Y, color_mat_W);
        }
        LOAD_MATRIX_18_SINGLE(tempEven, point_c, COLOR_MAT_Y);


        if (mu_positive){
          loadMatrixFromField(LINK_W, mymu, bc_link_nbr_idx, linkEven);
        }else{
          loadMatrixFromField(LINK_W, mymu, bc_link_nbr_idx, linkOdd);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, link_W);
        }

        if (mu_positive){    
          ADJ_MAT_MUL_MAT(link_W, color_mat_Y, link_X);
        }else{
          MAT_MUL_MAT(link_W, color_mat_Y, link_X);
        }

        if (sig_positive){
          loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkEven);
        }else{
          loadMatrixFromField(LINK_W, mysig, ab_link_nbr_idx, linkOdd);
        }
        if(ArrayLength<RealB>::compressed){
          RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, link_W);
        }

        if (sig_positive){        
          MAT_MUL_MAT(link_W, link_X, color_mat_Y);
        }else{
          ADJ_MAT_MUL_MAT(link_W, link_X, color_mat_Y);
        }

        const float & mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;
        if (sig_positive)
        {	
          MAT_MUL_MAT(link_X, color_mat_W, link_Z);
          ASSIGN_MAT(link_Z, color_mat_W);
          LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
          SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
          WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
        }

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


    template<class RealA, class RealB>
      static void
      all_link_kernel(const RealA* tempxEven, 
          const RealA* tempxOdd,
          RealA* QprevEven, 
          RealA* QprevOdd, 
          RealA* PmuEven, 
          RealA* PmuOdd,
          RealA* P3Even, 
          RealA* P3Odd,
          RealA* P3muEven, 
          RealA* P3muOdd,
          RealA* shortPEven, 
          RealA* shortPOdd,
          int sig, int mu,
          float coeff, float mcoeff, float accumu_coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          FullGauge cudaSiteLink,
          RealA* momEven, 
          RealA* momOdd,
          dim3 gridDim, dim3 blockDim,
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd)
          {
            dim3 halfGridDim(gridDim.x/2, 1,1);

            cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
            cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){		
              do_all_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( tempxEven,  
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
              do_all_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
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

              do_all_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( tempxEven,   
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

              do_all_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
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
              do_all_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( tempxEven,  
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


              do_all_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
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
              do_all_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( tempxEven, 
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

              do_all_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
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

            return;
          }



#define Pmu 	  tempmat[0]
#define P3        tempmat[1]
#define P5	  tempmat[2]
#define Pnumu     tempmat[3]
#define P3mu	  tempmat[3]
#define P5nu	  tempmat[3]
#define P7 	  tempmat[3]
#define Prhonumu  tempmat[3]
#define P7rho     tempmat[3]

// if first level of smearing
#define Qmu      tempCmat[0]
#define Qnumu    tempCmat[1]
#define Qrhonumu tempCmat[2] 
#define Q5       tempCmat[2]

    
    //template<class Real, class RealA, class RealB>
    template<class Real>
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


        int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
        dim3 blockDim(BLOCK_DIM,1,1);
        dim3 gridDim(volume/blockDim.x, 1, 1);


        for(sig=0; sig < 8; sig++){
          for(mu = 0; mu < 8; mu++){
            if ( (mu == sig) || (mu == OPP_DIR(sig))){
              continue;
            }
            //3-link
            //Kernel A: middle link

            int new_sig;
            if(GOES_BACKWARDS(sig)){ new_sig = OPP_DIR(sig); }else{ new_sig = sig; }

            middle_link_kernel( (float2*)cudaOprod.even.data[new_sig], (float2*)cudaOprod.odd.data[new_sig],
                (float2*)Pmu.even.data, (float2*)Pmu.odd.data,
                (float2*)P3.even.data, (float2*)P3.odd.data,
                (float2*)NULL,         (float2*)NULL,
                (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
                sig, mu, mThreeSt,
                (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
                gridDim, blockDim,
                (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); 

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
                  (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
                  (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
                  sig, nu, FiveSt,
                  (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
                  gridDim, blockDim,
                  (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); 

              checkCudaError();

              for(rho = 0; rho < 8; rho++){
                if (rho == sig || rho == OPP_DIR(sig)
                    || rho == mu || rho == OPP_DIR(mu)
                    || rho == nu || rho == OPP_DIR(nu)){
                  continue;
                }
                //7-link: middle link and side link
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


              //5-link: side link
              if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
              side_link_kernel((float2*)P5.even.data, (float2*)P5.odd.data,
                  (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
                  (float2*)P3.even.data, (float2*)P3.odd.data,
                  sig, nu, mFiveSt, coeff,
                  (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
                  (float2*)cudaMom.even, (float2*)cudaMom.odd,
                  gridDim, blockDim,
                  (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
              checkCudaError();



            } //nu 

            //lepage
            middle_link_kernel( (float2*)Pmu.even.data, (float2*)Pmu.odd.data,
                (float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
                (float2*)P5.even.data, (float2*)P5.odd.data,
                (float2*)Qmu.even.data, (float2*)Qmu.odd.data, 
                (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
                sig, mu, Lepage,
                (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
                gridDim, blockDim, 
                (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); 
            checkCudaError();		

            if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

            side_link_kernel((float2*)P5.even.data, (float2*)P5.odd.data,
                (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
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
                (float2*)NULL, (float2*)NULL,
                (float2*)NULL, (float2*)NULL,
                sig, mu, ThreeSt, coeff,
                (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
                (float2*)cudaMom.even, (float2*)cudaMom.odd,
                gridDim, blockDim,
                (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
            checkCudaError();			    

          }//mu
        }//sig



        for(sig=0; sig<8; ++sig){
          if(GOES_FORWARDS(sig)){
            one_and_naik_terms((float2*)cudaOprod.even.data[sig], (float2*)cudaOprod.odd.data[sig],
                sig, OneLink, 0.0,
                gridDim, blockDim,
                (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
          } // GOES_FORWARDS(sig)
          checkCudaError();
        }



        for(sig=0; sig<8; sig++){
          if(GOES_FORWARDS(sig)){
            compute_force_kernel( (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
                (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd,
                sig, gridDim, blockDim,
                (float2*)cudaMom.even, (float2*)cudaMom.odd);
          } // Only compute the force term if it goes forwards
        } // sig

        return; 
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
