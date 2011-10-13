#include <read_gauge.h>
#include <gauge_quda.h>

#include "hisq_force_quda.h"
#include "force_common.h"
#include "hw_quda.h"
#include "hisq_force_macros.h"

#include<utility>

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
  mat[0] = field[idx + dir*Vhx9];
  mat[1] = field[idx + dir*Vhx9 + Vh];
  mat[2] = field[idx + dir*Vhx9 + Vhx2];
  mat[3] = field[idx + dir*Vhx9 + Vhx3];
  mat[4] = field[idx + dir*Vhx9 + Vhx4];
  mat[5] = field[idx + dir*Vhx9 + Vhx5];
  mat[6] = field[idx + dir*Vhx9 + Vhx6];
  mat[7] = field[idx + dir*Vhx9 + Vhx7];
  mat[8] = field[idx + dir*Vhx9 + Vhx8];

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


    const float2 & operator+=(float2 & a, const float2 & b)
    {
      a.x += b.x;
      a.y += b.y;
    }

    const float4 & operator+=(float4 & a, const float4 & b)
    {
      a.x += b.x;
      a.y += b.y;
      a.z += b.z;
      a.w += b.w;
    }


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
          if( (i[3]&1)==1) *sign=-1;
          break;

        case YUP:
          if( ((i[3]+i[0])&1) == 1) *sign=-1; 
          break;

        case ZUP:
          if( ((i[3]+i[0]+i[1])&1) == 1) *sign=-1; 
          break;

        case TUP:
          if(i[3] == X4m1) *sign=-1; 
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

        RealA COLOR_MAT_W[ArrayLength<RealA>::result], COLOR_MAT_Y[ArrayLength<RealA>::result];

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
          const RealA* const QprevOdd, 		
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
        if(PmuOdd){
          WRITE_MATRIX_18_SINGLE(PmuOdd, point_b, COLOR_MAT_W);
        }

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
          //ASSIGN_MAT(link_Y, color_mat_W); 
          if(QmuEven){
            ASSIGN_MAT(link_Y, color_mat_W); 
            WRITE_MATRIX_18_SINGLE(QmuEven, sid, COLOR_MAT_W);
          }
        }else{ 
          LOAD_MATRIX_18_SINGLE(QprevOdd, point_d, COLOR_MAT_Y);   
          MAT_MUL_MAT(color_mat_Y, link_Y, color_mat_X);
          if(QmuEven){
            WRITE_MATRIX_18_SINGLE(QmuEven, sid, COLOR_MAT_X);
          }
          if(sig_positive){
            MAT_MUL_MAT(color_mat_W, color_mat_X, color_mat_Y);
          }	
        }


        if(sig_positive){
          const float & mycoeff = -CoeffSign<sig_positive,oddBit>::result*coeff;
          // Should be able to shorten this!
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
      middle_link_kernel(
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd,
          const RealA* const tempEven, 
          const RealA* const tempOdd, 
          RealA* const PmuEven, // write only  
          RealA* const PmuOdd, // write only
          RealA* const P3Even, // write only   
          RealA* const P3Odd,  // write only
          const RealA* const QprevEven, 
          const RealA* const QprevOdd,
          RealA* const QmuEven,  // write only
          RealA* const QmuOdd,   // write only
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          FullGauge cudaSiteLink,
          int sig, int mu, float coeff,
          dim3 gridDim, dim3 BlockDim)
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
      do_side_link_kernel(
          const RealA* const P3Even, 
          const RealA* const TempxEven, 
          const RealA* const TempxOdd,
          RealA* const shortPOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
          int sig, int mu, float coeff, float accumu_coeff,
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
          // Should be able to shorten this
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
          // Should be able to shorten this
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
        
          // Should be able to shorten this!
          LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
          SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, color_mat_X, mycoeff, color_mat_Z);
          WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
        }

        return;
      }




    template<class RealA, class RealB>
      static void
      side_link_kernel(
          RealA* momMatrixEven, 
          RealA* momMatrixOdd,
          const RealA* const P3Even, 
          const RealA* const P3Odd, 
          const RealA* const TempxEven, 
          const RealA* const TempxOdd,
          RealA* shortPEven,  
          RealA* shortPOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          FullGauge cudaSiteLink,
          int sig, int mu, float coeff, float accumu_coeff,
          dim3 gridDim, dim3 blockDim)
      {
        dim3 halfGridDim(gridDim.x/2,1,1);

        cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
        cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);   

        if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( P3Even, 
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven, linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_side_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);

        }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( P3Even, 
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven,  linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);		
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_side_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);		

        }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
          do_side_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( P3Even,
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven,  linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_side_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( P3Odd,
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);

        }else{
          do_side_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( P3Even,
              TempxEven,  TempxOdd,
              shortPOdd,
              linkEven, linkOdd,
              sig, mu, coeff, accumu_coeff,
              momMatrixEven, momMatrixOdd);
          cudaUnbindTexture(siteLink0TexSingle_recon);
          cudaUnbindTexture(siteLink1TexSingle_recon);

          //opposite binding
          cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
          cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

          do_side_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( P3Odd, 
              TempxOdd,  TempxEven,
              shortPEven,
              linkOdd, linkEven,
              sig, mu, coeff, accumu_coeff,
              momMatrixOdd, momMatrixEven);
        }

        cudaUnbindTexture(siteLink0TexSingle_recon);
        cudaUnbindTexture(siteLink1TexSingle_recon);    

        return;
      }


    template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
      __global__ void
      do_all_link_kernel(
          const RealA* const tempEven, 
          RealA* const QprevOdd,
          RealA* const shortPEven, 
          RealA* const shortPOdd,
          int sig, int mu, 
          float coeff, float accumu_coeff,
          const RealB* const linkEven, 
          const RealB* const linkOdd,
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

       // const RealB link_ptr[2]

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

        // I can define a new macro that does 
        // the multiplication of adjoint multiplication
        // depending on the mu_positive
        if (mu_positive){    
          ADJ_MAT_MUL_MAT(link_W, color_mat_Y, link_X);
        }else{
          MAT_MUL_MAT(link_W, color_mat_Y, link_X);
        }
        // I can use a pointer to the even and odd link fields 
        // to avoid all the if statements
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
          // Should be able to shorten this!
          LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z); 
          SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
          WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, COLOR_MAT_Z);
        }

        if (mu_positive)
        {
          MAT_MUL_MAT(color_mat_Y, color_mat_X, link_Z);
          // Should be able to shorten this!
          LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, COLOR_MAT_Z);
          SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
          WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, COLOR_MAT_Z);
          MAT_MUL_MAT(link_Y, color_mat_Y, color_mat_W);	
        }else
        {
          ADJ_MAT_MUL_ADJ_MAT(color_mat_X, color_mat_Y, link_Z);	
          // Should be able to shorten this!
          LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
          SCALAR_MULT_ADD_SU3_MATRIX(color_mat_Z, link_Z, mycoeff, color_mat_Z);
          WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, COLOR_MAT_Z);
          ADJ_MAT_MUL_MAT(link_Y, color_mat_Y, color_mat_W);	
        }
        
        // Should be able to shorten this!
        LOAD_MATRIX_18_SINGLE(shortPOdd, point_d, COLOR_MAT_Y);
        SCALAR_MULT_ADD_MATRIX(color_mat_Y, color_mat_W, accumu_coeff, color_mat_Y);
        WRITE_MATRIX_18_SINGLE(shortPOdd, point_d, COLOR_MAT_Y);

        return;
      }


    template<class RealA, class RealB>
      static void
      all_link_kernel(
          RealA* const momMatrixEven, 
          RealA* const momMatrixOdd,
          const RealA* const tempxEven, 
          const RealA* const tempxOdd,
          RealA* QprevEven, 
          RealA* QprevOdd, 
          RealA* shortPEven, 
          RealA* shortPOdd,
          const RealB* const linkEven, 
          const RealB* const linkOdd, 
          FullGauge cudaSiteLink,
          int sig, int mu,
          float coeff, float accumu_coeff,
          dim3 gridDim, dim3 blockDim)
          {
            dim3 halfGridDim(gridDim.x/2, 1,1);

            cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
            cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);

            if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){		
              do_all_link_kernel<RealA, RealB, 1, 1, 0><<<halfGridDim, blockDim>>>( tempxEven,  
                  QprevOdd, 
                  shortPEven,  shortPOdd,
                  sig,  mu,
                  coeff, accumu_coeff,
                  linkEven, linkOdd,
                  momMatrixEven, momMatrixOdd);
              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);
              do_all_link_kernel<RealA, RealB, 1, 1, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
                  QprevEven,
                  shortPOdd,  shortPEven,
                  sig,  mu,
                  coeff, accumu_coeff,
                  linkOdd, linkEven,
                  momMatrixOdd, momMatrixEven);	


            }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){

              do_all_link_kernel<RealA, RealB, 1, 0, 0><<<halfGridDim, blockDim>>>( tempxEven,   
                  QprevOdd,
                  shortPEven,  shortPOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkEven, linkOdd,
                  momMatrixEven, momMatrixOdd);	
              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

              do_all_link_kernel<RealA, RealB, 1, 0, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
                  QprevEven, 
                  shortPOdd,  shortPEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkOdd, linkEven,
                  momMatrixOdd, momMatrixEven);	

            }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
              do_all_link_kernel<RealA, RealB, 0, 1, 0><<<halfGridDim, blockDim>>>( tempxEven,  
                  QprevOdd, 
                  shortPEven,  shortPOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkEven, linkOdd,
                  momMatrixEven, momMatrixOdd);	
              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);


              do_all_link_kernel<RealA, RealB, 0, 1, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
                  QprevEven, 
                  shortPOdd,  shortPEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkOdd, linkEven,
                  momMatrixOdd, momMatrixEven);		
            }else{
              do_all_link_kernel<RealA, RealB, 0, 0, 0><<<halfGridDim, blockDim>>>( tempxEven, 
                  QprevOdd, 
                  shortPEven,  shortPOdd,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkEven, linkOdd,
                  momMatrixEven, momMatrixOdd);	

              cudaUnbindTexture(siteLink0TexSingle_recon);
              cudaUnbindTexture(siteLink1TexSingle_recon);

              //opposite binding
              cudaBindTexture(0, siteLink0TexSingle_recon, cudaSiteLink.odd, cudaSiteLink.bytes);
              cudaBindTexture(0, siteLink1TexSingle_recon, cudaSiteLink.even, cudaSiteLink.bytes);

              do_all_link_kernel<RealA, RealB, 0, 0, 1><<<halfGridDim, blockDim>>>( tempxOdd,  
                  QprevEven, 
                  shortPOdd,  shortPEven,
                  sig,  mu, 
                  coeff, accumu_coeff,
                  linkOdd, linkEven,
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

// if first level of smearing
#define Qmu      tempCmat[0]
#define Qnumu    tempCmat[1]

    
    //template<class Real, class RealA, class RealB>
    template<class Real, class  RealA, class RealB>
      static void
      do_hisq_force_cuda(Real eps, Real weight1, Real weight2,  Real* act_path_coeff, FullOprod cudaOprod, // need to change this code
          FullGauge cudaSiteLink, FullMom cudaMom, FullGauge cudaMomMatrix, FullMatrix tempmat[4], FullMatrix tempCmat[2], QudaGaugeParam* param)
      {

        Real coeff;

        Real OneLink, Lepage, Naik, FiveSt, ThreeSt, SevenSt;
        Real mLepage, mFiveSt, mThreeSt;

        Real ferm_epsilon;
        ferm_epsilon = 2.0*weight1*eps;
        OneLink = act_path_coeff[0]*ferm_epsilon ;
        Naik    = act_path_coeff[1]*ferm_epsilon ;
        ThreeSt = act_path_coeff[2]*ferm_epsilon ; mThreeSt = -ThreeSt;
        FiveSt  = act_path_coeff[3]*ferm_epsilon ; mFiveSt  = -FiveSt;
        SevenSt = act_path_coeff[4]*ferm_epsilon ; 
        Lepage  = act_path_coeff[5]*ferm_epsilon ; mLepage  = -Lepage;


        int volume = param->X[0]*param->X[1]*param->X[2]*param->X[3];
        dim3 blockDim(BLOCK_DIM,1,1);
        dim3 gridDim(volume/blockDim.x, 1, 1);


        for(int sig=0; sig<8; sig++){
          for(int mu=0; mu<8; mu++){
            if ( (mu == sig) || (mu == OPP_DIR(sig))){
              continue;
            }
            //3-link
            //Kernel A: middle link

            int new_sig;
            if(GOES_BACKWARDS(sig)){ new_sig = OPP_DIR(sig); }else{ new_sig = sig; }

            middle_link_kernel( 
                                (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                                (RealA*)cudaOprod.even.data[new_sig], (RealA*)cudaOprod.odd.data[new_sig], // read only
                                (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,                               // write only
                                (RealA*)P3.even.data, (RealA*)P3.odd.data,                                 // write only
                                (RealA*)NULL,         (RealA*)NULL,                                        // read only
                                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,                               // write only     
                                (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink,         // read only
                                sig, mu, mThreeSt,
                                gridDim, blockDim);

                                checkCudaError();

            for(int nu=0; nu < 8; nu++){
              if (nu == sig || nu == OPP_DIR(sig)
                  || nu == mu || nu == OPP_DIR(mu)){
                continue;
              }

              //5-link: middle link
              //Kernel B
              middle_link_kernel( 
                  (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                  (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,      // read only
                  (RealA*)Pnumu.even.data, (RealA*)Pnumu.odd.data,  // write only
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,        // write only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,      // read only
                  (RealA*)Qnumu.even.data, (RealA*)Qnumu.odd.data,  // write only
                  (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink, 
                  sig, nu, FiveSt,
                  gridDim, blockDim);

              checkCudaError();

              for(int rho = 0; rho < 8; rho++){
                if (rho == sig || rho == OPP_DIR(sig)
                    || rho == mu || rho == OPP_DIR(mu)
                    || rho == nu || rho == OPP_DIR(nu)){
                  continue;
                }
                //7-link: middle link and side link
                if(FiveSt != 0)coeff = SevenSt/FiveSt; else coeff = 0;
                all_link_kernel(
                    (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                    (RealA*)Pnumu.even.data, (RealA*)Pnumu.odd.data,
                    (RealA*)Qnumu.even.data, (RealA*)Qnumu.odd.data,
                    (RealA*)P5.even.data, (RealA*)P5.odd.data, // read and write
                    (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink,
                    sig, rho, SevenSt, coeff,
                    gridDim, blockDim);
                checkCudaError();

              }//rho  		


              //5-link: side link
              if(ThreeSt != 0)coeff = FiveSt/ThreeSt; else coeff = 0;
              side_link_kernel((RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                  (RealA*)P5.even.data, (RealA*)P5.odd.data,    // read only
                  (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,  // read only
                  (RealA*)P3.even.data, (RealA*)P3.odd.data,    // read and write
                  (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink,
                  sig, nu, mFiveSt, coeff,
                  gridDim, blockDim);
              checkCudaError();



            } //nu 

            //lepage
            middle_link_kernel( 
                (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                (RealA*)Pmu.even.data, (RealA*)Pmu.odd.data,     // read only
                (RealA*)NULL, (RealA*)NULL,                      // write only
                (RealA*)P5.even.data, (RealA*)P5.odd.data,       // write only
                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,     // read only
                (RealA*)NULL, (RealA*)NULL,                      // write only
                (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink, 
                sig, mu, Lepage,
                gridDim, blockDim);
            checkCudaError();		

            if(ThreeSt != 0)coeff = Lepage/ThreeSt ; else coeff = 0;

            side_link_kernel(
                (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                (RealA*)P5.even.data, (RealA*)P5.odd.data,           // read only
                (RealA*)Qmu.even.data, (RealA*)Qmu.odd.data,         // read only
                (RealA*)P3.even.data, (RealA*)P3.odd.data,           // read and write
                (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink,
                sig, mu, mLepage ,coeff,
                gridDim, blockDim);
            checkCudaError();		


            //3-link side link
            coeff=0.;
            side_link_kernel(
                (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                (RealA*)P3.even.data, (RealA*)P3.odd.data, // read only
                (RealA*)NULL, (RealA*)NULL,                // read only
                (RealA*)NULL, (RealA*)NULL,                // read and write
                (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink,
                sig, mu, ThreeSt, coeff,
                gridDim, blockDim);

            checkCudaError();			    

          }//mu
        }//sig



        for(int sig=0; sig<8; ++sig){
          if(GOES_FORWARDS(sig)){
            one_and_naik_terms((RealA*)cudaOprod.even.data[sig], (RealA*)cudaOprod.odd.data[sig],
                sig, OneLink, 0.0,
                gridDim, blockDim,
                (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd);
          } // GOES_FORWARDS(sig)
          checkCudaError();
        }



        for(int sig=0; sig<8; sig++){
          if(GOES_FORWARDS(sig)){
            compute_force_kernel( (RealB*)cudaSiteLink.even, (RealB*)cudaSiteLink.odd, cudaSiteLink,
                (RealA*)cudaMomMatrix.even, (RealA*)cudaMomMatrix.odd,
                sig, gridDim, blockDim,
                (RealA*)cudaMom.even, (RealA*)cudaMom.odd);
          } // Only compute the force term if it goes forwards
        } // sig

        return; 
      }

#undef Pmu
#undef Pnumu
#undef P3
#undef P5

#undef Qmu
#undef Qnumu


    void
      hisq_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
          FullOprod cudaOprod, FullGauge cudaSiteLink, FullMom cudaMom, FullGauge cudaMomMatrix, QudaGaugeParam* param)
      {

        FullMatrix tempmat[4];
        for(int i=0; i<4; i++){
          tempmat[i]  = createMatQuda(param->X, param->cuda_prec);
        }

        FullMatrix tempCompmat[2];
        for(int i=0; i<2; i++){
          tempCompmat[i] = createMatQuda(param->X, param->cuda_prec);
        }	


        if (param->cuda_prec == QUDA_DOUBLE_PRECISION){
        }else{	
          do_hisq_force_cuda<float,float2,float2>( (float)eps, (float)weight1, (float)weight2, (float*)act_path_coeff,
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
