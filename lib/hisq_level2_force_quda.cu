#include <read_gauge.h>
#include <gauge_quda.h>

#include "hisq_force_quda.h"
#include "force_common.h"
#include "hw_quda.h"


// The following code computes the contribution 
// of level2 hisq smearing to the fermion force. 
// Well, in fact, right now, the routines 
// below return a momentum, whereas they should 
// actually return a color-matrix, which is 
// later combined with contributions from the
// link unitarization and level1 smearing to 
// give the net force.



#define LOAD_ANTI_HERMITIAN LOAD_ANTI_HERMITIAN_SINGLE

#define LOAD_HW_SINGLE(hw, idx, var)	do{	      \
	var##0 = hw[idx + 0*Vh];		      \
	var##1 = hw[idx + 1*Vh];		      \
	var##2 = hw[idx + 2*Vh];		      \
	var##3 = hw[idx + 3*Vh];		      \
	var##4 = hw[idx + 4*Vh];		      \
	var##5 = hw[idx + 5*Vh];		      \
    }while(0)


#define WRITE_HW_SINGLE(hw, idx, var)	do{				\
	hw[idx + 0*Vh] = var##0;					\
	hw[idx + 1*Vh] = var##1;					\
	hw[idx + 2*Vh] = var##2;					\
	hw[idx + 3*Vh] = var##3;					\
	hw[idx + 4*Vh] = var##4;					\
	hw[idx + 5*Vh] = var##5;					\
    }while(0)

#define LOAD_HW(hw, idx, var) LOAD_HW_SINGLE(hw, idx, var)
#define WRITE_HW(hw, idx, var) WRITE_HW_SINGLE(hw, idx, var)
#define LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)
//#define LOAD_ANTI_HERMITIAN(mom, mydir, idx, AH);

#define FF_SITE_MATRIX_LOAD_TEX 1

#if (FF_SITE_MATRIX_LOAD_TEX == 1)
#define linkEvenTex siteLink0TexSingle
#define linkOddTex siteLink1TexSingle
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE_TEX(src##Tex, dir, idx, var)
#else
#define FF_LOAD_MATRIX(src, dir, idx, var) LOAD_MATRIX_12_SINGLE(src, dir, idx, var)
#endif

// The following macros are needed for the hisq routines:

// Load the outer product from memory
// This is an array of 3x3 complex matrices. 
// These matrices are not unitary, and so they 
// are stored as 9 complex numbers
/*
#define LOAD_MATRIX_18_SINGLE(gauge, dir, idx, var)		\
  var##0 = gauge[idx + dir*Vhx9];				\
  var##1 = gauge[idx + dir*Vhx9 + Vh];				\
  var##2 = gauge[idx + dir*Vhx9 + Vhx2];			\
  var##3 = gauge[idx + dir*Vhx9 + Vhx3];			\
  var##4 = gauge[idx + dir*Vhx9 + Vhx4];			\
  var##5 = gauge[idx + dir*Vhx9 + Vhx5];			\
  var##6 = gauge[idx + dir*Vhx9 + Vhx6];			\
  var##7 = gauge[idx + dir*Vhx9 + Vhx7];			\
  var##8 = gauge[idx + dir*Vhx9 + Vhx8];
*/


#define LOAD_MOM_MATRIX_SINGLE(mom, dir, idx, var)	\
  var##0 = mom[idx + dir*Vhx9];				\
  var##1 = mom[idx + dir*Vhx9 + Vh];			\
  var##2 = mom[idx + dir*Vhx9 + Vhx2];			\
  var##3 = mom[idx + dir*Vhx9 + Vhx3];			\
  var##4 = mom[idx + dir*Vhx9 + Vhx4];			\
  var##5 = mom[idx + dir*Vhx9 + Vhx5];			\
  var##6 = mom[idx + dir*Vhx9 + Vhx6];			\
  var##7 = mom[idx + dir*Vhx9 + Vhx7];			\
  var##8 = mom[idx + dir*Vhx9 + Vhx8];




#define LOAD_MATRIX_18_SINGLE(gauge, idx, var)			\
  var##0 = gauge[idx];						\
  var##1 = gauge[idx + Vh];					\
  var##2 = gauge[idx + Vhx2];					\
  var##3 = gauge[idx + Vhx3];					\
  var##4 = gauge[idx + Vhx4];					\
  var##5 = gauge[idx + Vhx5];					\
  var##6 = gauge[idx + Vhx6];					\
  var##7 = gauge[idx + Vhx7];					\
  var##8 = gauge[idx + Vhx8];



// Write to an array of float2 
#define WRITE_MATRIX_18_SINGLE(mat, idx, var) do{ \
 	mat[idx + 0*Vh] = var##0;  \
	mat[idx + 1*Vh] = var##1;  \
	mat[idx + 2*Vh] = var##2;  \
	mat[idx + 3*Vh] = var##3;  \
	mat[idx + 4*Vh] = var##4;  \
	mat[idx + 5*Vh] = var##5;  \
	mat[idx + 6*Vh] = var##6;  \
	mat[idx + 7*Vh] = var##7;  \
	mat[idx + 8*Vh] = var##8;  \
}while(0)

#define WRITE_MOM_MATRIX_SINGLE(mat, dir, idx, var) do{ \
 	mat[idx + dir*Vhx9 + 0*Vh] = var##0;  \
	mat[idx + dir*Vhx9 + 1*Vh] = var##1;  \
	mat[idx + dir*Vhx9 + 2*Vh] = var##2;  \
	mat[idx + dir*Vhx9 + 3*Vh] = var##3;  \
	mat[idx + dir*Vhx9 + 4*Vh] = var##4;  \
	mat[idx + dir*Vhx9 + 5*Vh] = var##5;  \
	mat[idx + dir*Vhx9 + 6*Vh] = var##6;  \
	mat[idx + dir*Vhx9 + 7*Vh] = var##7;  \
	mat[idx + dir*Vhx9 + 8*Vh] = var##8;  \
}while(0)


#define ACCUMULATE_MOM_MATRIX_SINGLE(mat, dir, idx, var) do{ \
        mat[idx + dir*Vhx9 + 0*Vh] = var##0;  \
        mat[idx + dir*Vhx9 + 1*Vh] = var##1;  \
        mat[idx + dir*Vhx9 + 2*Vh] = var##2;  \
        mat[idx + dir*Vhx9 + 3*Vh] = var##3;  \
        mat[idx + dir*Vhx9 + 4*Vh] = var##4;  \
        mat[idx + dir*Vhx9 + 5*Vh] = var##5;  \
        mat[idx + dir*Vhx9 + 6*Vh] = var##6;  \
        mat[idx + dir*Vhx9 + 7*Vh] = var##7;  \
        mat[idx + dir*Vhx9 + 8*Vh] = var##8;  \
}while(0)


// Write to an array of float4 
#define WRITE_MATRIX_12_SINGLE(mat, idx, var) do{ \
	mat[idx + 0*Vh] = var##0;  \
	mat[idx + 1*Vh] = var##1;  \
	mat[idx + 2*Vh] = var##2;  \
}while(0)


#define oprod00_re OPROD0.x
#define oprod00_im OPROD0.y
#define oprod01_re OPROD1.x
#define oprod01_im OPROD1.y
#define oprod02_re OPROD2.x
#define oprod02_im OPROD2.y
#define oprod10_re OPROD3.x
#define oprod10_im OPROD3.y 
#define oprod11_re OPROD4.x
#define oprod11_im OPROD4.y
#define oprod12_re OPROD5.x
#define oprod12_im OPROD5.y
#define oprod20_re OPROD6.x
#define oprod20_im OPROD6.y
#define oprod21_re OPROD7.x
#define oprod21_im OPROD7.y
#define oprod22_re OPROD8.x
#define oprod22_im OPROD8.y


#define linkd00_re LINKD0.x
#define linkd00_im LINKD0.y
#define linkd01_re LINKD1.x
#define linkd01_im LINKD1.y
#define linkd02_re LINKD2.x
#define linkd02_im LINKD2.y
#define linkd10_re LINKD3.x
#define linkd10_im LINKD3.y 
#define linkd11_re LINKD4.x
#define linkd11_im LINKD4.y
#define linkd12_re LINKD5.x
#define linkd12_im LINKD5.y
#define linkd20_re LINKD6.x
#define linkd20_im LINKD6.y
#define linkd21_re LINKD7.x
#define linkd21_im LINKD7.y
#define linkd22_re LINKD8.x
#define linkd22_im LINKD8.y



#define temp00_re TEMP0.x
#define temp00_im TEMP0.y
#define temp01_re TEMP0.z
#define temp01_im TEMP0.w
#define temp02_re TEMP1.x
#define temp02_im TEMP1.y
#define temp10_re TEMP1.z
#define temp10_im TEMP1.w
#define temp11_re TEMP2.x
#define temp11_im TEMP2.y
#define temp12_re TEMP2.z
#define temp12_im TEMP2.w
#define temp20_re TEMP3.x
#define temp20_im TEMP3.y
#define temp21_re TEMP3.z
#define temp21_im TEMP3.w
#define temp22_re TEMP4.x
#define temp22_im TEMP4.y





#define linkab00_re LINKAB0.x
#define linkab00_im LINKAB0.y
#define linkab01_re LINKAB1.x
#define linkab01_im LINKAB1.y
#define linkab02_re LINKAB2.x
#define linkab02_im LINKAB2.y
#define linkab10_re LINKAB3.x
#define linkab10_im LINKAB3.y 
#define linkab11_re LINKAB4.x
#define linkab11_im LINKAB4.y
#define linkab12_re LINKAB5.x
#define linkab12_im LINKAB5.y
#define linkab20_re LINKAB6.x
#define linkab20_im LINKAB6.y
#define linkab21_re LINKAB7.x
#define linkab21_im LINKAB7.y
#define linkab22_re LINKAB8.x
#define linkab22_im LINKAB8.y


#define store00_re STORE0.x
#define store00_im STORE0.y
#define store01_re STORE1.x
#define store01_im STORE1.y
#define store02_re STORE2.x
#define store02_im STORE2.y
#define store10_re STORE3.x
#define store10_im STORE3.y 
#define store11_re STORE4.x
#define store11_im STORE4.y
#define store12_re STORE5.x
#define store12_im STORE5.y
#define store20_re STORE6.x
#define store20_im STORE6.y
#define store21_re STORE7.x
#define store21_im STORE7.y
#define store22_re STORE8.x
#define store22_im STORE8.y




#define ADJ_MAT(a, b) \
 b##00_re =  a##00_re; \
 b##00_im = -a##00_im; \
 b##01_re =  a##10_re; \
 b##01_im = -a##10_im; \
 b##02_re =  a##20_re; \
 b##02_im = -a##20_im; \
 b##10_re =  a##01_re; \
 b##10_im = -a##01_im; \
 b##11_re =  a##11_re; \
 b##11_im = -a##11_im; \
 b##12_re =  a##21_re; \
 b##12_im = -a##21_im; \
 b##20_re =  a##02_re; \
 b##20_im = -a##02_im; \
 b##21_re =  a##12_re; \
 b##21_im = -a##12_im; \
 b##22_re =  a##22_re; \
 b##22_im = -a##22_im; 


#define ASSIGN_MAT(a, b) \
 b##00_re =  a##00_re; \
 b##00_im =  a##00_im; \
 b##01_re =  a##01_re; \
 b##01_im =  a##01_im; \
 b##02_re =  a##02_re; \
 b##02_im =  a##02_im; \
 b##10_re =  a##10_re; \
 b##10_im =  a##10_im; \
 b##11_re =  a##11_re; \
 b##11_im =  a##11_im; \
 b##12_re =  a##12_re; \
 b##12_im =  a##12_im; \
 b##20_re =  a##20_re; \
 b##20_im =  a##20_im; \
 b##21_re =  a##21_re; \
 b##21_im =  a##21_im; \
 b##22_re =  a##22_re; \
 b##22_im =  a##22_im; \






#define SET_IDENTITY(b) \
 b##00_re =  1; \
 b##00_im =  0; \
 b##01_re =  0; \
 b##01_im =  0; \
 b##02_re =  0; \
 b##02_im =  0; \
 b##10_re =  0; \
 b##10_im =  0; \
 b##11_re =  1; \
 b##11_im =  0; \
 b##12_re =  0; \
 b##12_im =  0; \
 b##20_re =  0; \
 b##20_im =  0; \
 b##21_re =  0; \
 b##21_im =  0; \
 b##22_re =  1; \
 b##22_im =  0; 



#define MAT_MUL_MAT(a, b, c) \
 c##00_re = a##00_re*b##00_re - a##00_im*b##00_im + a##01_re*b##10_re - a##01_im*b##10_im + a##02_re*b##20_re - a##02_im*b##20_im; \
 c##00_im = a##00_re*b##00_im + a##00_im*b##00_re + a##01_re*b##10_im + a##01_im*b##10_re + a##02_re*b##20_im + a##02_im*b##20_re; \
 c##01_re = a##00_re*b##01_re - a##00_im*b##01_im + a##01_re*b##11_re - a##01_im*b##11_im + a##02_re*b##21_re - a##02_im*b##21_im; \
 c##01_im = a##00_re*b##01_im + a##00_im*b##01_re + a##01_re*b##11_im + a##01_im*b##11_re + a##02_re*b##21_im + a##02_im*b##21_re; \
 c##02_re = a##00_re*b##02_re - a##00_im*b##02_im + a##01_re*b##12_re - a##01_im*b##12_im + a##02_re*b##22_re - a##02_im*b##22_im; \
 c##02_im = a##00_re*b##02_im + a##00_im*b##02_re + a##01_re*b##12_im + a##01_im*b##12_re + a##02_re*b##22_im + a##02_im*b##22_re; \
 c##10_re = a##10_re*b##00_re - a##10_im*b##00_im + a##11_re*b##10_re - a##11_im*b##10_im + a##12_re*b##20_re - a##12_im*b##20_im; \
 c##10_im = a##10_re*b##00_im + a##10_im*b##00_re + a##11_re*b##10_im + a##11_im*b##10_re + a##12_re*b##20_im + a##12_im*b##20_re; \
 c##11_re = a##10_re*b##01_re - a##10_im*b##01_im + a##11_re*b##11_re - a##11_im*b##11_im + a##12_re*b##21_re - a##12_im*b##21_im; \
 c##11_im = a##10_re*b##01_im + a##10_im*b##01_re + a##11_re*b##11_im + a##11_im*b##11_re + a##12_re*b##21_im + a##12_im*b##21_re; \
 c##12_re = a##10_re*b##02_re - a##10_im*b##02_im + a##11_re*b##12_re - a##11_im*b##12_im + a##12_re*b##22_re - a##12_im*b##22_im; \
 c##12_im = a##10_re*b##02_im + a##10_im*b##02_re + a##11_re*b##12_im + a##11_im*b##12_re + a##12_re*b##22_im + a##12_im*b##22_re; \
 c##20_re = a##20_re*b##00_re - a##20_im*b##00_im + a##21_re*b##10_re - a##21_im*b##10_im + a##22_re*b##20_re - a##22_im*b##20_im; \
 c##20_im = a##20_re*b##00_im + a##20_im*b##00_re + a##21_re*b##10_im + a##21_im*b##10_re + a##22_re*b##20_im + a##22_im*b##20_re; \
 c##21_re = a##20_re*b##01_re - a##20_im*b##01_im + a##21_re*b##11_re - a##21_im*b##11_im + a##22_re*b##21_re - a##22_im*b##21_im; \
 c##21_im = a##20_re*b##01_im + a##20_im*b##01_re + a##21_re*b##11_im + a##21_im*b##11_re + a##22_re*b##21_im + a##22_im*b##21_re; \
 c##22_re = a##20_re*b##02_re - a##20_im*b##02_im + a##21_re*b##12_re - a##21_im*b##12_im + a##22_re*b##22_re - a##22_im*b##22_im; \
 c##22_im = a##20_re*b##02_im + a##20_im*b##02_re + a##21_re*b##12_im + a##21_im*b##12_re + a##22_re*b##22_im + a##22_im*b##22_re; 
 
#define MAT_MUL_ADJ_MAT(a, b, c) \
 c##00_re =    a##00_re*b##00_re + a##00_im*b##00_im + a##01_re*b##01_re + a##01_im*b##01_im + a##02_re*b##02_re + a##02_im*b##02_im; \
 c##00_im =  - a##00_re*b##00_im + a##00_im*b##00_re - a##01_re*b##01_im + a##01_im*b##01_re - a##02_re*b##02_im + a##02_im*b##02_re; \
 c##01_re =    a##00_re*b##10_re + a##00_im*b##10_im + a##01_re*b##11_re + a##01_im*b##11_im + a##02_re*b##12_re + a##02_im*b##12_im; \
 c##01_im =  - a##00_re*b##10_im + a##00_im*b##10_re - a##01_re*b##11_im + a##01_im*b##11_re - a##02_re*b##12_im + a##02_im*b##12_re; \
 c##02_re =    a##00_re*b##20_re + a##00_im*b##20_im + a##01_re*b##21_re + a##01_im*b##21_im + a##02_re*b##22_re + a##02_im*b##22_im; \
 c##02_im =  - a##00_re*b##20_im + a##00_im*b##20_re - a##01_re*b##21_im + a##01_im*b##21_re - a##02_re*b##22_im + a##02_im*b##22_re; \
 c##10_re =    a##10_re*b##00_re + a##10_im*b##00_im + a##11_re*b##01_re + a##11_im*b##01_im + a##12_re*b##02_re + a##12_im*b##02_im; \
 c##10_im =  - a##10_re*b##00_im + a##10_im*b##00_re - a##11_re*b##01_im + a##11_im*b##01_re - a##12_re*b##02_im + a##12_im*b##02_re; \
 c##11_re =    a##10_re*b##10_re + a##10_im*b##10_im + a##11_re*b##11_re + a##11_im*b##11_im + a##12_re*b##12_re + a##12_im*b##12_im; \
 c##11_im =  - a##10_re*b##10_im + a##10_im*b##10_re - a##11_re*b##11_im + a##11_im*b##11_re - a##12_re*b##12_im + a##12_im*b##12_re; \
 c##12_re =    a##10_re*b##20_re + a##10_im*b##20_im + a##11_re*b##21_re + a##11_im*b##21_im + a##12_re*b##22_re + a##12_im*b##22_im; \
 c##12_im =  - a##10_re*b##20_im + a##10_im*b##20_re - a##11_re*b##21_im + a##11_im*b##21_re - a##12_re*b##22_im + a##12_im*b##22_re; \
 c##20_re =    a##20_re*b##00_re + a##20_im*b##00_im + a##21_re*b##01_re + a##21_im*b##01_im + a##22_re*b##02_re + a##22_im*b##02_im; \
 c##20_im =  - a##20_re*b##00_im + a##20_im*b##00_re - a##21_re*b##01_im + a##21_im*b##01_re - a##22_re*b##02_im + a##22_im*b##02_re; \
 c##21_re =    a##20_re*b##10_re + a##20_im*b##10_im + a##21_re*b##11_re + a##21_im*b##11_im + a##22_re*b##12_re + a##22_im*b##12_im; \
 c##21_im =  - a##20_re*b##10_im + a##20_im*b##10_re - a##21_re*b##11_im + a##21_im*b##11_re - a##22_re*b##12_im + a##22_im*b##12_re; \
 c##22_re =    a##20_re*b##20_re + a##20_im*b##20_im + a##21_re*b##21_re + a##21_im*b##21_im + a##22_re*b##22_re + a##22_im*b##22_im; \
 c##22_im =  - a##20_re*b##20_im + a##20_im*b##20_re - a##21_re*b##21_im + a##21_im*b##21_re - a##22_re*b##22_im + a##22_im*b##22_re; 
 
#define ADJ_MAT_MUL_MAT(a, b, c) \
 c##00_re = a##00_re*b##00_re + a##00_im*b##00_im + a##10_re*b##10_re + a##10_im*b##10_im + a##20_re*b##20_re + a##20_im*b##20_im; \
 c##00_im = a##00_re*b##00_im - a##00_im*b##00_re + a##10_re*b##10_im - a##10_im*b##10_re + a##20_re*b##20_im - a##20_im*b##20_re; \
 c##01_re = a##00_re*b##01_re + a##00_im*b##01_im + a##10_re*b##11_re + a##10_im*b##11_im + a##20_re*b##21_re + a##20_im*b##21_im; \
 c##01_im = a##00_re*b##01_im - a##00_im*b##01_re + a##10_re*b##11_im - a##10_im*b##11_re + a##20_re*b##21_im - a##20_im*b##21_re; \
 c##02_re = a##00_re*b##02_re + a##00_im*b##02_im + a##10_re*b##12_re + a##10_im*b##12_im + a##20_re*b##22_re + a##20_im*b##22_im; \
 c##02_im = a##00_re*b##02_im - a##00_im*b##02_re + a##10_re*b##12_im - a##10_im*b##12_re + a##20_re*b##22_im - a##20_im*b##22_re; \
 c##10_re = a##01_re*b##00_re + a##01_im*b##00_im + a##11_re*b##10_re + a##11_im*b##10_im + a##21_re*b##20_re + a##21_im*b##20_im; \
 c##10_im = a##01_re*b##00_im - a##01_im*b##00_re + a##11_re*b##10_im - a##11_im*b##10_re + a##21_re*b##20_im - a##21_im*b##20_re; \
 c##11_re = a##01_re*b##01_re + a##01_im*b##01_im + a##11_re*b##11_re + a##11_im*b##11_im + a##21_re*b##21_re + a##21_im*b##21_im; \
 c##11_im = a##01_re*b##01_im - a##01_im*b##01_re + a##11_re*b##11_im - a##11_im*b##11_re + a##21_re*b##21_im - a##21_im*b##21_re; \
 c##12_re = a##01_re*b##02_re + a##01_im*b##02_im + a##11_re*b##12_re + a##11_im*b##12_im + a##21_re*b##22_re + a##21_im*b##22_im; \
 c##12_im = a##01_re*b##02_im - a##01_im*b##02_re + a##11_re*b##12_im - a##11_im*b##12_re + a##21_re*b##22_im - a##21_im*b##22_re; \
 c##20_re = a##02_re*b##00_re + a##02_im*b##00_im + a##12_re*b##10_re + a##12_im*b##10_im + a##22_re*b##20_re + a##22_im*b##20_im; \
 c##20_im = a##02_re*b##00_im - a##02_im*b##00_re + a##12_re*b##10_im - a##12_im*b##10_re + a##22_re*b##20_im - a##22_im*b##20_re; \
 c##21_re = a##02_re*b##01_re + a##02_im*b##01_im + a##12_re*b##11_re + a##12_im*b##11_im + a##22_re*b##21_re + a##22_im*b##21_im; \
 c##21_im = a##02_re*b##01_im - a##02_im*b##01_re + a##12_re*b##11_im - a##12_im*b##11_re + a##22_re*b##21_im - a##22_im*b##21_re; \
 c##22_re = a##02_re*b##02_re + a##02_im*b##02_im + a##12_re*b##12_re + a##12_im*b##12_im + a##22_re*b##22_re + a##22_im*b##22_im; \
 c##22_im = a##02_re*b##02_im - a##02_im*b##02_re + a##12_re*b##12_im - a##12_im*b##12_re + a##22_re*b##22_im - a##22_im*b##22_re; 

#define ADJ_MAT_MUL_ADJ_MAT(a, b, c) \
 c##00_re =    a##00_re*b##00_re - a##00_im*b##00_im + a##10_re*b##01_re - a##10_im*b##01_im + a##20_re*b##02_re - a##20_im*b##02_im; \
 c##00_im =  - a##00_re*b##00_im - a##00_im*b##00_re - a##10_re*b##01_im - a##10_im*b##01_re - a##20_re*b##02_im - a##20_im*b##02_re; \
 c##01_re =    a##00_re*b##10_re - a##00_im*b##10_im + a##10_re*b##11_re - a##10_im*b##11_im + a##20_re*b##12_re - a##20_im*b##12_im; \
 c##01_im =  - a##00_re*b##10_im - a##00_im*b##10_re - a##10_re*b##11_im - a##10_im*b##11_re - a##20_re*b##12_im - a##20_im*b##12_re; \
 c##02_re =    a##00_re*b##20_re - a##00_im*b##20_im + a##10_re*b##21_re - a##10_im*b##21_im + a##20_re*b##22_re - a##20_im*b##22_im; \
 c##02_im =  - a##00_re*b##20_im - a##00_im*b##20_re - a##10_re*b##21_im - a##10_im*b##21_re - a##20_re*b##22_im - a##20_im*b##22_re; \
 c##10_re =    a##01_re*b##00_re - a##01_im*b##00_im + a##11_re*b##01_re - a##11_im*b##01_im + a##21_re*b##02_re - a##21_im*b##02_im; \
 c##10_im =  - a##01_re*b##00_im - a##01_im*b##00_re - a##11_re*b##01_im - a##11_im*b##01_re - a##21_re*b##02_im - a##21_im*b##02_re; \
 c##11_re =    a##01_re*b##10_re - a##01_im*b##10_im + a##11_re*b##11_re - a##11_im*b##11_im + a##21_re*b##12_re - a##21_im*b##12_im; \
 c##11_im =  - a##01_re*b##10_im - a##01_im*b##10_re - a##11_re*b##11_im - a##11_im*b##11_re - a##21_re*b##12_im - a##21_im*b##12_re; \
 c##12_re =    a##01_re*b##20_re - a##01_im*b##20_im + a##11_re*b##21_re - a##11_im*b##21_im + a##21_re*b##22_re - a##21_im*b##22_im; \
 c##12_im =  - a##01_re*b##20_im - a##01_im*b##20_re - a##11_re*b##21_im - a##11_im*b##21_re - a##21_re*b##22_im - a##21_im*b##22_re; \
 c##20_re =    a##02_re*b##00_re - a##02_im*b##00_im + a##12_re*b##01_re - a##12_im*b##01_im + a##22_re*b##02_re - a##22_im*b##02_im; \
 c##20_im =  - a##02_re*b##00_im - a##02_im*b##00_re - a##12_re*b##01_im - a##12_im*b##01_re - a##22_re*b##02_im - a##22_im*b##02_re; \
 c##21_re =    a##02_re*b##10_re - a##02_im*b##10_im + a##12_re*b##11_re - a##12_im*b##11_im + a##22_re*b##12_re - a##22_im*b##12_im; \
 c##21_im =  - a##02_re*b##10_im - a##02_im*b##10_re - a##12_re*b##11_im - a##12_im*b##11_re - a##22_re*b##12_im - a##22_im*b##12_re; \
 c##22_re =    a##02_re*b##20_re - a##02_im*b##20_im + a##12_re*b##21_re - a##12_im*b##21_im + a##22_re*b##22_re - a##22_im*b##22_im; \
 c##22_im =  - a##02_re*b##20_im - a##02_im*b##20_re - a##12_re*b##21_im - a##12_im*b##21_re - a##22_re*b##22_im - a##22_im*b##22_re; 

// end of macros specific to hisq routines

#define MAT_MUL_HW(M, HW, HWOUT)					\
    HWOUT##00_re = (M##00_re * HW##00_re - M##00_im * HW##00_im)	\
	+          (M##01_re * HW##01_re - M##01_im * HW##01_im)	\
	+          (M##02_re * HW##02_re - M##02_im * HW##02_im);	\
    HWOUT##00_im = (M##00_re * HW##00_im + M##00_im * HW##00_re)	\
	+          (M##01_re * HW##01_im + M##01_im * HW##01_re)	\
	+          (M##02_re * HW##02_im + M##02_im * HW##02_re);	\
    HWOUT##01_re = (M##10_re * HW##00_re - M##10_im * HW##00_im)	\
	+          (M##11_re * HW##01_re - M##11_im * HW##01_im)	\
	+          (M##12_re * HW##02_re - M##12_im * HW##02_im);	\
    HWOUT##01_im = (M##10_re * HW##00_im + M##10_im * HW##00_re) 	\
	+          (M##11_re * HW##01_im + M##11_im * HW##01_re)	\
	+          (M##12_re * HW##02_im + M##12_im * HW##02_re);	\
    HWOUT##02_re = (M##20_re * HW##00_re - M##20_im * HW##00_im)	\
	+          (M##21_re * HW##01_re - M##21_im * HW##01_im)	\
	+          (M##22_re * HW##02_re - M##22_im * HW##02_im);	\
    HWOUT##02_im = (M##20_re * HW##00_im + M##20_im * HW##00_re)	\
	+          (M##21_re * HW##01_im + M##21_im * HW##01_re)	\
	+          (M##22_re * HW##02_im + M##22_im * HW##02_re);	\
    HWOUT##10_re = (M##00_re * HW##10_re - M##00_im * HW##10_im)	\
	+          (M##01_re * HW##11_re - M##01_im * HW##11_im)	\
	+          (M##02_re * HW##12_re - M##02_im * HW##12_im);	\
    HWOUT##10_im = (M##00_re * HW##10_im + M##00_im * HW##10_re)	\
	+          (M##01_re * HW##11_im + M##01_im * HW##11_re)	\
	+          (M##02_re * HW##12_im + M##02_im * HW##12_re);	\
    HWOUT##11_re = (M##10_re * HW##10_re - M##10_im * HW##10_im)	\
	+          (M##11_re * HW##11_re - M##11_im * HW##11_im)	\
	+          (M##12_re * HW##12_re - M##12_im * HW##12_im);	\
    HWOUT##11_im = (M##10_re * HW##10_im + M##10_im * HW##10_re) 	\
	+          (M##11_re * HW##11_im + M##11_im * HW##11_re)	\
	+          (M##12_re * HW##12_im + M##12_im * HW##12_re);	\
    HWOUT##12_re = (M##20_re * HW##10_re - M##20_im * HW##10_im)	\
	+          (M##21_re * HW##11_re - M##21_im * HW##11_im)	\
	+          (M##22_re * HW##12_re - M##22_im * HW##12_im);	\
    HWOUT##12_im = (M##20_re * HW##10_im + M##20_im * HW##10_re)	\
	+          (M##21_re * HW##11_im + M##21_im * HW##11_re)	\
	+          (M##22_re * HW##12_im + M##22_im * HW##12_re);


#define ADJ_MAT_MUL_HW(M, HW, HWOUT)					\
    HWOUT##00_re = (M##00_re * HW##00_re + M##00_im * HW##00_im)	\
	+          (M##10_re * HW##01_re + M##10_im * HW##01_im)	\
	+          (M##20_re * HW##02_re + M##20_im * HW##02_im);	\
    HWOUT##00_im = (M##00_re * HW##00_im - M##00_im * HW##00_re)	\
	+          (M##10_re * HW##01_im - M##10_im * HW##01_re)	\
	+          (M##20_re * HW##02_im - M##20_im * HW##02_re);	\
    HWOUT##01_re = (M##01_re * HW##00_re + M##01_im * HW##00_im)	\
	+          (M##11_re * HW##01_re + M##11_im * HW##01_im)	\
	+          (M##21_re * HW##02_re + M##21_im * HW##02_im);	\
    HWOUT##01_im = (M##01_re * HW##00_im - M##01_im * HW##00_re)	\
	+          (M##11_re * HW##01_im - M##11_im * HW##01_re)	\
	+          (M##21_re * HW##02_im - M##21_im * HW##02_re);	\
    HWOUT##02_re = (M##02_re * HW##00_re + M##02_im * HW##00_im)	\
	+          (M##12_re * HW##01_re + M##12_im * HW##01_im)	\
	+          (M##22_re * HW##02_re + M##22_im * HW##02_im);	\
    HWOUT##02_im = (M##02_re * HW##00_im - M##02_im * HW##00_re)	\
	+          (M##12_re * HW##01_im - M##12_im * HW##01_re)	\
	+          (M##22_re * HW##02_im - M##22_im * HW##02_re);	\
    HWOUT##10_re = (M##00_re * HW##10_re + M##00_im * HW##10_im)	\
	+          (M##10_re * HW##11_re + M##10_im * HW##11_im)	\
	+          (M##20_re * HW##12_re + M##20_im * HW##12_im);	\
    HWOUT##10_im = (M##00_re * HW##10_im - M##00_im * HW##10_re)	\
	+          (M##10_re * HW##11_im - M##10_im * HW##11_re)	\
	+          (M##20_re * HW##12_im - M##20_im * HW##12_re);	\
    HWOUT##11_re = (M##01_re * HW##10_re + M##01_im * HW##10_im)	\
	+          (M##11_re * HW##11_re + M##11_im * HW##11_im)	\
	+          (M##21_re * HW##12_re + M##21_im * HW##12_im);	\
    HWOUT##11_im = (M##01_re * HW##10_im - M##01_im * HW##10_re)	\
	+          (M##11_re * HW##11_im - M##11_im * HW##11_re)	\
	+          (M##21_re * HW##12_im - M##21_im * HW##12_re);	\
    HWOUT##12_re = (M##02_re * HW##10_re + M##02_im * HW##10_im)	\
	+          (M##12_re * HW##11_re + M##12_im * HW##11_im)	\
	+          (M##22_re * HW##12_re + M##22_im * HW##12_im);	\
    HWOUT##12_im = (M##02_re * HW##10_im - M##02_im * HW##10_re)	\
	+          (M##12_re * HW##11_im - M##12_im * HW##11_re)	\
	+          (M##22_re * HW##12_im - M##22_im * HW##12_re);


#define SU3_PROJECTOR(va, vb, m)					\
    m##00_re = va##0_re * vb##0_re + va##0_im * vb##0_im;		\
    m##00_im = va##0_im * vb##0_re - va##0_re * vb##0_im;		\
    m##01_re = va##0_re * vb##1_re + va##0_im * vb##1_im;		\
    m##01_im = va##0_im * vb##1_re - va##0_re * vb##1_im;		\
    m##02_re = va##0_re * vb##2_re + va##0_im * vb##2_im;		\
    m##02_im = va##0_im * vb##2_re - va##0_re * vb##2_im;		\
    m##10_re = va##1_re * vb##0_re + va##1_im * vb##0_im;		\
    m##10_im = va##1_im * vb##0_re - va##1_re * vb##0_im;		\
    m##11_re = va##1_re * vb##1_re + va##1_im * vb##1_im;		\
    m##11_im = va##1_im * vb##1_re - va##1_re * vb##1_im;		\
    m##12_re = va##1_re * vb##2_re + va##1_im * vb##2_im;		\
    m##12_im = va##1_im * vb##2_re - va##1_re * vb##2_im;		\
    m##20_re = va##2_re * vb##0_re + va##2_im * vb##0_im;		\
    m##20_im = va##2_im * vb##0_re - va##2_re * vb##0_im;		\
    m##21_re = va##2_re * vb##1_re + va##2_im * vb##1_im;		\
    m##21_im = va##2_im * vb##1_re - va##2_re * vb##1_im;		\
    m##22_re = va##2_re * vb##2_re + va##2_im * vb##2_im;		\
    m##22_im = va##2_im * vb##2_re - va##2_re * vb##2_im;

//vc = va + vb*s 
#define SCALAR_MULT_ADD_SU3_VECTOR(va, vb, s, vc) do {	\
	vc##0_re = va##0_re + vb##0_re * s;		\
	vc##0_im = va##0_im + vb##0_im * s;		\
	vc##1_re = va##1_re + vb##1_re * s;		\
	vc##1_im = va##1_im + vb##1_im * s;		\
	vc##2_re = va##2_re + vb##2_re * s;		\
	vc##2_im = va##2_im + vb##2_im * s;		\
    }while (0)


#define SCALAR_MULT_ADD_MATRIX(a, b, scalar, c) do{ \
  c##00_re = a##00_re + scalar*b##00_re;  \
  c##00_im = a##00_im + scalar*b##00_im;  \
  c##01_re = a##01_re + scalar*b##01_re;  \
  c##01_im = a##01_im + scalar*b##01_im;  \
  c##02_re = a##02_re + scalar*b##02_re;  \
  c##02_im = a##02_im + scalar*b##02_im;  \
  c##10_re = a##10_re + scalar*b##10_re;  \
  c##10_im = a##10_im + scalar*b##10_im;  \
  c##11_re = a##11_re + scalar*b##11_re;  \
  c##11_im = a##11_im + scalar*b##11_im;  \
  c##12_re = a##12_re + scalar*b##12_re;  \
  c##12_im = a##12_im + scalar*b##12_im;  \
  c##20_re = a##20_re + scalar*b##20_re;  \
  c##20_im = a##20_im + scalar*b##20_im;  \
  c##21_re = a##21_re + scalar*b##21_re;  \
  c##21_im = a##21_im + scalar*b##21_im;  \
  c##22_re = a##22_re + scalar*b##22_re;  \
  c##22_im = a##22_im + scalar*b##22_im;  \
}while(0)


#define SCALAR_MULT_MATRIX(scalar, b, c) do{ \
  c##00_re = scalar*b##00_re;  \
  c##00_im = scalar*b##00_im;  \
  c##01_re = scalar*b##01_re;  \
  c##01_im = scalar*b##01_im;  \
  c##02_re = scalar*b##02_re;  \
  c##02_im = scalar*b##02_im;  \
  c##10_re = scalar*b##10_re;  \
  c##10_im = scalar*b##10_im;  \
  c##11_re = scalar*b##11_re;  \
  c##11_im = scalar*b##11_im;  \
  c##12_re = scalar*b##12_re;  \
  c##12_im = scalar*b##12_im;  \
  c##20_re = scalar*b##20_re;  \
  c##20_im = scalar*b##20_im;  \
  c##21_re = scalar*b##21_re;  \
  c##21_im = scalar*b##21_im;  \
  c##22_re = scalar*b##22_re;  \
  c##22_im = scalar*b##22_im;  \
}while(0)



#define FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mydir, idx, new_idx) do {	\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_idx = ( (new_x1==X1m1)?idx-X1m1:idx+1);			\
            new_x1 = (new_x1==X1m1)?0:new_x1+1;                         \
            break;                                                      \
        case 1:                                                         \
            new_idx = ( (new_x2==X2m1)?idx-X2X1mX1:idx+X1);		\
            new_x2 = (new_x2==X2m1)?0:new_x2+1;                         \
            break;                                                      \
        case 2:                                                         \
            new_idx = ( (new_x3==X3m1)?idx-X3X2X1mX2X1:idx+X2X1);	\
            new_x3 = (new_x3==X3m1)?0:new_x3+1;                         \
            break;                                                      \
        case 3:                                                         \
            new_idx = ( (new_x4==X4m1)?idx-X4X3X2X1mX3X2X1:idx+X3X2X1); \
            new_x4 = (new_x4==X4m1)?0:new_x4+1;                         \
            break;                                                      \
        }                                                               \
    }while(0)

#define FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mydir, idx, new_idx) do {	\
        switch(mydir){                                                  \
        case 0:                                                         \
            new_idx = ( (new_x1==0)?idx+X1m1:idx-1);			\
            new_x1 = (new_x1==0)?X1m1:new_x1 - 1;                       \
            break;                                                      \
        case 1:                                                         \
            new_idx = ( (new_x2==0)?idx+X2X1mX1:idx-X1);		\
            new_x2 = (new_x2==0)?X2m1:new_x2 - 1;                       \
            break;                                                      \
        case 2:                                                         \
            new_idx = ( (new_x3==0)?idx+X3X2X1mX2X1:idx-X2X1);		\
            new_x3 = (new_x3==0)?X3m1:new_x3 - 1;                       \
            break;                                                      \
        case 3:                                                         \
            new_idx = ( (new_x4==0)?idx+X4X3X2X1mX3X2X1:idx-X3X2X1);	\
            new_x4 = (new_x4==0)?X4m1:new_x4 - 1;                       \
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

//this macro require linka, linkb, and ah variables defined
#define ADD_FORCE_TO_MOM(hw1, hw2, mom, idx, dir, cf,oddness) do{	\
	float my_coeff;						\
	int mydir;							\
	if (GOES_BACKWARDS(dir)){					\
	    mydir=OPP_DIR(dir);						\
	    my_coeff = -cf;						\
	}else{								\
	    mydir=dir;							\
	    my_coeff = cf;						\
	}								\
	float tmp_coeff;						\
	tmp_coeff = my_coeff;					        \
	if(oddness){							\
	    tmp_coeff = - my_coeff;					\
	}								\
	LOAD_ANTI_HERMITIAN(mom, mydir, idx, AH);			\
	UNCOMPRESS_ANTI_HERMITIAN(ah, linka);				\
	SU3_PROJECTOR(hw1##0, hw2##0, linkb);				\
	SCALAR_MULT_ADD_SU3_MATRIX(linka, linkb, tmp_coeff, linka);	\
	SU3_PROJECTOR(hw1##1, hw2##1, linkb);				\
	SCALAR_MULT_ADD_SU3_MATRIX(linka, linkb, tmp_coeff, linka);	\
	MAKE_ANTI_HERMITIAN(linka, ah);					\
	WRITE_ANTI_HERMITIAN_SINGLE(mom, mydir, idx, AH);		\
    }while(0)



//this macro require linka, linkb and ah variables defined
#define ADD_MAT_FORCE_TO_MOM(mat1, mat2, mom, idx, dir, cf,oddness) do{	\
	float my_coeff;							\
	int mydir;							\
	if (GOES_BACKWARDS(dir)){					\
	    mydir=OPP_DIR(dir);						\
	    my_coeff = -cf;						\
	}else{								\
	    mydir=dir;							\
	    my_coeff = cf;						\
	}								\
	float tmp_coeff;						\
	tmp_coeff = my_coeff;						\
	if(oddness){							\
	    tmp_coeff = - my_coeff;					\
	}								\
	LOAD_ANTI_HERMITIAN(mom, mydir, idx, AH);			\
	UNCOMPRESS_ANTI_HERMITIAN(ah, linka);				\
	MAT_MUL_ADJ_MAT(mat1, mat2, linkb)					\
	SCALAR_MULT_ADD_SU3_MATRIX(linka, linkb, tmp_coeff, linka);	\
	MAKE_ANTI_HERMITIAN(linka, ah);					\
	WRITE_ANTI_HERMITIAN_SINGLE(mom, mydir, idx, AH);		\
    }while(0)



//this macro require linka, linkb and ah variables defined
#define MAT_FORCE_TO_MOM(mat, mom, idx, dir, cf,oddness) do{	\
	float my_coeff;							\
	int mydir;							\
	if (GOES_BACKWARDS(dir)){					\
	    mydir=OPP_DIR(dir);						\
	    my_coeff = -cf;						\
	}else{								\
	    mydir=dir;							\
	    my_coeff = cf;						\
	}								\
	float tmp_coeff;						\
	tmp_coeff = my_coeff;						\
	if(oddness){							\
	    tmp_coeff = - my_coeff;					\
	}								\
	LOAD_ANTI_HERMITIAN(mom, mydir, idx, AH);			\
	UNCOMPRESS_ANTI_HERMITIAN(ah, linka);				\
	SCALAR_MULT_ADD_SU3_MATRIX(linka, mat, tmp_coeff, linka);	\
	MAKE_ANTI_HERMITIAN(linka, ah);					\
	WRITE_ANTI_HERMITIAN_SINGLE(mom, mydir, idx, AH);		\
    }while(0)


#define SIMPLE_MAT_FORCE_TO_MOM(mat, mom, idx, dir) do{		\
	LOAD_ANTI_HERMITIAN(mom, dir, idx, AH);			\
	UNCOMPRESS_ANTI_HERMITIAN(ah, linka);			\
	SCALAR_MULT_ADD_SU3_MATRIX(linka, mat, 1.0, linka);	\
	MAKE_ANTI_HERMITIAN(linka, ah);				\
	WRITE_ANTI_HERMITIAN_SINGLE(mom, dir, idx, AH);		\
    }while(0)







#define FF_COMPUTE_RECONSTRUCT_SIGN(sign, dir, i1,i2,i3,i4) do {        \
        sign =1;                                                        \
        switch(dir){                                                    \
        case XUP:                                                       \
            if ( (i4 & 1) == 1){                                        \
                sign = -1;                                              \
            }                                                           \
            break;                                                      \
        case YUP:                                                       \
            if ( ((i4+i1) & 1) == 1){                                   \
                sign = -1;                                              \
            }                                                           \
            break;                                                      \
        case ZUP:                                                       \
            if ( ((i4+i1+i2) & 1) == 1){                                \
                sign = -1;                                              \
            }                                                           \
            break;                                                      \
        case TUP:                                                       \
            if (i4 == X4m1 ){                                           \
                sign = -1;                                              \
            }                                                           \
            break;                                                      \
        }                                                               \
    }while (0)


#define hwa00_re HWA0.x
#define hwa00_im HWA0.y
#define hwa01_re HWA1.x
#define hwa01_im HWA1.y
#define hwa02_re HWA2.x
#define hwa02_im HWA2.y
#define hwa10_re HWA3.x
#define hwa10_im HWA3.y
#define hwa11_re HWA4.x
#define hwa11_im HWA4.y
#define hwa12_re HWA5.x
#define hwa12_im HWA5.y

#define hwb00_re HWB0.x
#define hwb00_im HWB0.y
#define hwb01_re HWB1.x
#define hwb01_im HWB1.y
#define hwb02_re HWB2.x
#define hwb02_im HWB2.y
#define hwb10_re HWB3.x
#define hwb10_im HWB3.y
#define hwb11_re HWB4.x
#define hwb11_im HWB4.y
#define hwb12_re HWB5.x
#define hwb12_im HWB5.y

#define hwc00_re HWC0.x
#define hwc00_im HWC0.y
#define hwc01_re HWC1.x
#define hwc01_im HWC1.y
#define hwc02_re HWC2.x
#define hwc02_im HWC2.y
#define hwc10_re HWC3.x
#define hwc10_im HWC3.y
#define hwc11_re HWC4.x
#define hwc11_im HWC4.y
#define hwc12_re HWC5.x
#define hwc12_im HWC5.y

#define hwd00_re HWD0.x
#define hwd00_im HWD0.y
#define hwd01_re HWD1.x
#define hwd01_im HWD1.y
#define hwd02_re HWD2.x
#define hwd02_im HWD2.y
#define hwd10_re HWD3.x
#define hwd10_im HWD3.y
#define hwd11_re HWD4.x
#define hwd11_im HWD4.y
#define hwd12_re HWD5.x
#define hwd12_im HWD5.y

#define hwe00_re HWE0.x
#define hwe00_im HWE0.y
#define hwe01_re HWE1.x
#define hwe01_im HWE1.y
#define hwe02_re HWE2.x
#define hwe02_im HWE2.y
#define hwe10_re HWE3.x
#define hwe10_im HWE3.y
#define hwe11_re HWE4.x
#define hwe11_im HWE4.y
#define hwe12_re HWE5.x
#define hwe12_im HWE5.y


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

    int z1 = FAST_INT_DIVIDE(sid, X1h);
    int x1h = sid - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    int X = 2*sid + x1odd;

    int link_sign;

    float2 AH0, AH1, AH2, AH3, AH4;
    float4 LINKA0, LINKA1, LINKA2, LINKA3, LINKA4; // 10 complex numbers
    float2 STORE0, STORE1, STORE2, STORE3, STORE4, STORE5, STORE6, STORE7, STORE8; 
    float2 LINKD0, LINKD1, LINKD2, LINKD3, LINKD4, LINKD5, LINKD6, LINKD7, LINKD8;

    FF_LOAD_MATRIX(linkEven, sig, sid, LINKA);
    FF_COMPUTE_RECONSTRUCT_SIGN(link_sign, sig, x1, x2, x3, x4);	
    RECONSTRUCT_LINK_12(sig, sid, link_sign, linka);

    LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);
    MAT_MUL_MAT(linka, store, linkd);
    SIMPLE_MAT_FORCE_TO_MOM(linkd, momEven, sid, sig);
}








template<int sig_positive, int mu_positive, int oddBit> 
__global__ void
do_middle_link_kernel(float2* tempxxEven, 
		      float2* PmuOdd, float2* P3Even,
		      float2* QprevOdd, 		
		      float2* QmuEven, float2* Q3Even,
		      int sig, int mu, int lambda, float coeff,
		      float4* linkEven, float4* linkOdd,
		      float2* momEven, bool threeStaple,
                      float2* momMatrixEven // Just added this in!
		      ) // are we on a threeStaple?
{							 // pointer to integer used to reconstruct Qprev
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z1 = FAST_INT_DIVIDE(sid, X1h);
    int x1h = sid - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    int X = 2*sid + x1odd;

    int new_x1, new_x2, new_x3, new_x4;
    int new_mem_idx;
    int ad_link_sign=1;
    int ab_link_sign=1;
    int bc_link_sign=1;
    
    float4 LINKA0, LINKA1, LINKA2, LINKA3, LINKA4; // 10 complex numbers
    float4 LINKB0, LINKB1, LINKB2, LINKB3, LINKB4; // 10 complex numbers
    float4 LINKC0, LINKC1, LINKC2, LINKC3, LINKC4; // 10 complex numbers

    float4 TEMP0, TEMP1, TEMP2, TEMP3, TEMP4; // need to find a way to get around this

    float2 LINKD0, LINKD1, LINKD2, LINKD3, LINKD4, LINKD5, LINKD6, LINKD7, LINKD8;
    float2 OPROD0, OPROD1, OPROD2, OPROD3, OPROD4, OPROD5, OPROD6, OPROD7, OPROD8; // 9 complex numbers
										   // The outer product is not stored in a compact form

    float2 LINKAB0, LINKAB1, LINKAB2, LINKAB3, LINKAB4, LINKAB5, LINKAB6, LINKAB7, LINKAB8;
    float2 STORE0, STORE1, STORE2, STORE3, STORE4, STORE5, STORE6, STORE7, STORE8; 


    float2 AH0, AH1, AH2, AH3, AH4;
  
   //        A________B
   //    mu   |      |
   // 	    D |      |C
   //	  
   //	  A is the current point (sid)
    int point_b, point_c, point_d;
    int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
    int mymu;
 
    new_x1 = x1;
    new_x2 = x2;
    new_x3 = x3;
    new_x4 = x4;
    
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
	FF_COMPUTE_RECONSTRUCT_SIGN(ad_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }else{
	ad_link_nbr_idx = sid;
	FF_COMPUTE_RECONSTRUCT_SIGN(ad_link_sign, mymu, x1, x2, x3, x4);	
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
	FF_COMPUTE_RECONSTRUCT_SIGN(bc_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }
    // So far, we have just computed ad_link_nbr_idx and 
    // bc_link_nbr_idx
	
    new_x1 = x1;
    new_x2 = x2;
    new_x3 = x3;
    new_x4 = x4;
    if(sig_positive){
	FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
    }else{
	FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
    }
    point_b = (new_mem_idx >> 1); 
    
    if (!mu_positive){
	bc_link_nbr_idx = point_b;
	FF_COMPUTE_RECONSTRUCT_SIGN(bc_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }   
    
    if(sig_positive){
	ab_link_nbr_idx = sid;
	FF_COMPUTE_RECONSTRUCT_SIGN(ab_link_sign, mysig, x1, x2, x3, x4);	
    }else{	
	ab_link_nbr_idx = point_b;
	FF_COMPUTE_RECONSTRUCT_SIGN(ab_link_sign, mysig, new_x1,new_x2,new_x3,new_x4);
    }
    // now we have ab_link_nbr_idx
    
  
    // load the link variable connecting a and b 
    // Store in linka 
    if(sig_positive){
      FF_LOAD_MATRIX(linkEven, mysig, ab_link_nbr_idx, LINKA);	
    }else{
      FF_LOAD_MATRIX(linkOdd, mysig, ab_link_nbr_idx, LINKA);	
    }
    RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, linka);
 
    // load the link variable connecting b and c 
    // Store in linkb
    if(mu_positive){
      FF_LOAD_MATRIX(linkEven, mymu, bc_link_nbr_idx, LINKB);
    }else{ 
      FF_LOAD_MATRIX(linkOdd, mymu, bc_link_nbr_idx, LINKB);	
    }
    RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, linkb);


   LOAD_MATRIX_18_SINGLE(tempxxEven, point_c, OPROD);
    // I do not think that Q3 is needed!
    if(mu_positive){
      ADJ_MAT_MUL_MAT(linkb, oprod, linkd);
    }else{
      MAT_MUL_MAT(linkb, oprod, linkd);
    }
    // Why write to PmuOdd instead of PmuEven?
    // Well, PmuEven would require tempxxOdd 
    // i.e., an extra device-memory access
    WRITE_MATRIX_18_SINGLE(PmuOdd, point_b, LINKD);
    if(sig_positive){
	MAT_MUL_MAT(linka, linkd, oprod);
    }else{ 
	ADJ_MAT_MUL_MAT(linka, linkd, oprod);
    }
    WRITE_MATRIX_18_SINGLE(P3Even, sid, OPROD);


    if(mu_positive){
      FF_LOAD_MATRIX(linkOdd, mymu, ad_link_nbr_idx, LINKC);
      RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, linkc);
    }else{
      FF_LOAD_MATRIX(linkEven, mymu, ad_link_nbr_idx, LINKB);
      RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, linkb);
      ADJ_MAT(linkb, linkc);
    }
   


    float mycoeff; 

    if(threeStaple){
      if(sig_positive){
        MAT_MUL_MAT(linkd, linkc, oprod);

	if(oddBit==1){mycoeff = -coeff;}
	else{mycoeff = coeff;}
     
        

	// These lines are important!
        LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);
	SCALAR_MULT_ADD_SU3_MATRIX(store, oprod, mycoeff, store);
        WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);


/*
        SCALAR_MULT_MATRIX(mycoeff, oprod, oprod);
	WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, OPROD);
        LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, OPROD); 

        MAT_MUL_MAT(linka, oprod, linkd);
	SIMPLE_MAT_FORCE_TO_MOM(linkd, momEven, sid, sig);
*/
        //MAT_FORCE_TO_MOM(linkd, momEven, sid, sig, coeff, oddBit);

      }
      ASSIGN_MAT(linkc, linkd);
      WRITE_MATRIX_18_SINGLE(QmuEven, sid, LINKD);
    }else{ // !threeStaple
      LOAD_MATRIX_18_SINGLE(QprevOdd, point_d, LINKAB);
      MAT_MUL_MAT(linkab, linkc, temp);
      ASSIGN_MAT(temp, linkab);
      WRITE_MATRIX_18_SINGLE(QmuEven, sid, LINKAB);

      if(sig_positive){
        MAT_MUL_MAT(linkd, linkab, oprod);

	if(oddBit==1){mycoeff = -coeff;}
	else{mycoeff = coeff;}

	LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);
	SCALAR_MULT_ADD_SU3_MATRIX(store, oprod, mycoeff, store);
	WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);

/*
        SCALAR_MULT_MATRIX(mycoeff, oprod, oprod);
	WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, OPROD);
        LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, OPROD); 

        MAT_MUL_MAT(linka, oprod, linkd);

	SIMPLE_MAT_FORCE_TO_MOM(linkd, momEven, sid, sig);
        //MAT_FORCE_TO_MOM(linkd, momEven, sid, sig, coeff, oddBit);   

*/
      }	
    }
}



static void 
compute_force_kernel(float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
		     float2* momMatrixEven, float2* momMatrixOdd,
		     int sig, dim3 gridDim, dim3 blockDim,
		     float2* momEven, float2* momOdd)
{
  dim3 halfGridDim(gridDim.x/2, 1, 1);
  
  // Need to see if this is necessary in the lates version of quda
  cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);
  cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.odd,  cudaSiteLink.bytes);

  do_compute_force_kernel<0><<<halfGridDim, blockDim>>>(linkEven, linkOdd,
							momMatrixEven, momMatrixOdd,
							sig, 
							momEven, momOdd);
  cudaUnbindTexture(siteLink0TexSingle);
  cudaUnbindTexture(siteLink1TexSingle);


  do_compute_force_kernel<1><<<halfGridDim, blockDim>>>(linkOdd, linkEven,
						       momMatrixOdd, momMatrixEven,
						       sig,
						       momOdd, momEven);

  cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
  cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

  cudaUnbindTexture(siteLink0TexSingle);
  cudaUnbindTexture(siteLink1TexSingle);

}





static void
middle_link_kernel(float2* tempxxEven, float2* tempxxOdd, 
		   float2* PmuEven, float2* PmuOdd,
		   float2* P3Even, float2* P3Odd,
		   float2* QprevEven, float2* QprevOdd,
		   float2* QmuEven, float2* QmuOdd,
		   float2* Q3Even, float2* Q3Odd,
		   int sig, int mu, int lambda, float coeff,
		   float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
		   float2* momEven, float2* momOdd,
		   dim3 gridDim, dim3 BlockDim, bool threeStaple, 
		   float2* momMatrixEven, float2* momMatrixOdd)
{
    dim3 halfGridDim(gridDim.x/2, 1,1);

    cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
   
    if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){	
	do_middle_link_kernel<1,1,0><<<halfGridDim, BlockDim>>>( tempxxEven,
								 PmuOdd,  P3Even,
								 QprevOdd,
								 QmuEven,  Q3Even,
								 sig, mu, lambda, coeff,
								 linkEven, linkOdd,
								 momEven, threeStaple,
								 momMatrixEven);
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);
	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_middle_link_kernel<1,1,1><<<halfGridDim, BlockDim>>>( tempxxOdd, 
								 PmuEven,  P3Odd,
								 QprevEven,
								 QmuOdd,  Q3Odd,
								 sig, mu, lambda, coeff,
								 linkOdd, linkEven,
								 momOdd, threeStaple,
								 momMatrixOdd);
    }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	do_middle_link_kernel<1,0,0><<<halfGridDim, BlockDim>>>( tempxxEven,
								 PmuOdd,  P3Even,
								 QprevOdd,
								 QmuEven,  Q3Even,
								 sig, mu, lambda, coeff,
								 linkEven, linkOdd,
								 momEven, threeStaple,
								 momMatrixEven);	
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_middle_link_kernel<1,0,1><<<halfGridDim, BlockDim>>>( tempxxOdd, 
								 PmuEven,  P3Odd,
								 QprevEven,
								 QmuOdd,  Q3Odd,
								 sig, mu, lambda, coeff,
								 linkOdd, linkEven,
								 momOdd, threeStaple,
								 momMatrixOdd);
	
    }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	do_middle_link_kernel<0,1,0><<<halfGridDim, BlockDim>>>( tempxxEven, 
								 PmuOdd,  P3Even,
								 QprevOdd,
								 QmuEven,  Q3Even,
								 sig, mu, lambda, coeff,
								 linkEven, linkOdd,
								 momEven, threeStaple,	
								 momMatrixEven);	
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_middle_link_kernel<0,1,1><<<halfGridDim, BlockDim>>>( tempxxOdd,
								 PmuEven,  P3Odd,
								 QprevEven, 
								 QmuOdd,  Q3Odd,
								 sig, mu, lambda, coeff,
								 linkOdd, linkEven,
								 momOdd, threeStaple,
								 momMatrixOdd);
    }else{
	do_middle_link_kernel<0,0,0><<<halfGridDim, BlockDim>>>( tempxxEven,
								 PmuOdd, P3Even,
								 QprevOdd,
								 QmuEven, Q3Even,
								 sig, mu, lambda, coeff,
								 linkEven, linkOdd,
								 momEven, threeStaple,
								 momMatrixEven);		

	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_middle_link_kernel<0,0,1><<<halfGridDim, BlockDim>>>( tempxxOdd, 
								 PmuEven,  P3Odd,
								 QprevEven,
								 QmuOdd,  Q3Odd,
								 sig, mu, lambda, coeff,
								 linkOdd, linkEven,
								 momOdd, threeStaple,
								 momMatrixOdd);		
    }
    cudaUnbindTexture(siteLink0TexSingle);
    cudaUnbindTexture(siteLink1TexSingle);    
}


template<int sig_positive, int mu_positive, int oddBit>
__global__ void
do_side_link_kernel(float2* P3Even, float2* P3Odd, 
		 float2* P3muEven, float2* P3muOdd,
		 float2* TempxEven, float2* TempxOdd,
		 float2* QmuEven,  float2* QmuOdd,
		 float2* shortPEven,  float2* shortPOdd,
		 int sig, int mu, float coeff, float accumu_coeff,
		 float4* linkEven, float4* linkOdd,
		 float2* momEven, float2* momOdd,
		 float2* momMatrixEven, float2* momMatrixOdd)
{
    float mcoeff;
    mcoeff = -coeff;
    
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int z1 = FAST_INT_DIVIDE(sid, X1h);
    int x1h = sid - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    int X = 2*sid + x1odd;

    int ad_link_sign = 1;
    float4 LINKA0, LINKA1, LINKA2, LINKA3, LINKA4;
    float4 LINKB0, LINKB1, LINKB2, LINKB3, LINKB4;
    
    float2 AH0, AH1, AH2, AH3, AH4;

    float2 STORE0, STORE1, STORE2, STORE3, STORE4, STORE5, STORE6, STORE7, STORE8;
    float2 LINKAB0, LINKAB1, LINKAB2, LINKAB3, LINKAB4, LINKAB5, LINKAB6, LINKAB7, LINKAB8; 
    float2 LINKD0, LINKD1, LINKD2, LINKD3, LINKD4, LINKD5, LINKD6, LINKD7, LINKD8;
    float2 OPROD0, OPROD1, OPROD2, OPROD3, OPROD4, OPROD5, OPROD6, OPROD7, OPROD8;
    
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

    int new_x1 = x1;
    int new_x2 = x2;
    int new_x3 = x3;
    int new_x4 = x4;

    if(mu_positive){
	mymu =mu;
	FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mymu,X, new_mem_idx);
    }else{
	mymu = OPP_DIR(mu);
	FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mymu, X, new_mem_idx);
    }
    point_d = (new_mem_idx >> 1);


    if (mu_positive){
	ad_link_nbr_idx = point_d;
	FF_COMPUTE_RECONSTRUCT_SIGN(ad_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }else{
	ad_link_nbr_idx = sid;
	FF_COMPUTE_RECONSTRUCT_SIGN(ad_link_sign, mymu, x1, x2, x3, x4);	
    }

    
    LOAD_MATRIX_18_SINGLE(P3Even, sid, OPROD);
    if(mu_positive){
	FF_LOAD_MATRIX(linkOdd, mymu, ad_link_nbr_idx, LINKA);
    }else{
	FF_LOAD_MATRIX(linkEven, mymu, ad_link_nbr_idx, LINKA);
    }

    RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, linka);	


    // Should all be inside if (shortPOdd)
    if (shortPOdd){
        if (mu_positive){
	  MAT_MUL_MAT(linka, oprod, linkd);
        }else{
	  ADJ_MAT_MUL_MAT(linka, oprod, linkd);
        }
        LOAD_MATRIX_18_SINGLE(shortPOdd, point_d, LINKAB);
        SCALAR_MULT_ADD_MATRIX(linkab, linkd, accumu_coeff, linkab);
        WRITE_MATRIX_18_SINGLE(shortPOdd, point_d, LINKAB);
    }

    
    //start to add side link force
    if (mu_positive){
	if(TempxOdd){
	  LOAD_MATRIX_18_SINGLE(TempxOdd, point_d, LINKAB);
          MAT_MUL_MAT(oprod, linkab, linkd);
	}else{
	  ASSIGN_MAT(oprod, linkd);
	}
	if(sig_positive){
	  if(oddBit==1){mycoeff = coeff;}
	  else{mycoeff = -coeff;}
	}else{
	  if(oddBit==1){mycoeff = -coeff;}
	  else{mycoeff = coeff;}
	}

        LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, STORE);
        SCALAR_MULT_ADD_SU3_MATRIX(store, linkd, mycoeff, store);
        WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, STORE);

/*
        SCALAR_MULT_MATRIX(mycoeff, linkd, linkd);
	WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, LINKD);
        LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, LINKD); 
	MAT_MUL_MAT(linka, linkd, oprod);
	SIMPLE_MAT_FORCE_TO_MOM(oprod, momOdd, point_d, mu);
*/

    }else{

	if(TempxOdd){
          LOAD_MATRIX_18_SINGLE(TempxOdd, point_d, LINKAB);
        }else{
	  SET_IDENTITY(linkab);
	}
        ADJ_MAT(linkab,linkd);
	MAT_MUL_ADJ_MAT(linkd, oprod, linkab);
        
        if(sig_positive){ 
	  if(oddBit==1){mycoeff = coeff;}
	  else{mycoeff = -coeff;}
        }else{
	  if(oddBit==1){mycoeff = -coeff;}
	  else{mycoeff = coeff;}
        }


        LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, STORE);
        SCALAR_MULT_ADD_SU3_MATRIX(store, linkab, mycoeff, store);
        WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, STORE);

/*
	SCALAR_MULT_MATRIX(mycoeff, linkab, linkab);	
        WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, LINKAB);
	LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, LINKAB);
	MAT_MUL_MAT(linka, linkab, linkb);
	// Note that since mu goes backwards, OPP_DIR(mu) goes forwards.
	SIMPLE_MAT_FORCE_TO_MOM(linkb, momEven, sid, OPP_DIR(mu));	    
*/
    }

}





static void
side_link_kernel(float2* P3Even, float2* P3Odd, 
		 float2* P3muEven, float2* P3muOdd,
		 float2* TempxEven, float2* TempxOdd,
		 float2* QmuEven,  float2* QmuOdd,
		 float2* shortPEven,  float2* shortPOdd,
		 int sig, int mu, float coeff, float accumu_coeff,
		 float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
		 float2* momEven, float2* momOdd,
		 dim3 gridDim, dim3 blockDim,
		 float2* momMatrixEven, float2* momMatrixOdd)
{
    dim3 halfGridDim(gridDim.x/2,1,1);
    
    cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);   

    if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){
	do_side_link_kernel<1,1,0><<<halfGridDim, blockDim>>>( P3Even,  P3Odd, 
							       P3muEven,  P3muOdd,
							       TempxEven,  TempxOdd,
							       QmuEven,   QmuOdd,
							       shortPEven,   shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven, linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_side_link_kernel<1,1,1><<<halfGridDim, blockDim>>>( P3Odd,  P3Even, 
							       P3muOdd,  P3muEven,
							       TempxOdd,  TempxEven,
							       QmuOdd,   QmuEven,
							       shortPOdd,   shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);
	
    }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){
	do_side_link_kernel<1,0,0><<<halfGridDim, blockDim>>>( P3Even,  P3Odd, 
							       P3muEven,  P3muOdd,
							       TempxEven,  TempxOdd,
							       QmuEven,   QmuOdd,
							       shortPEven,   shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven,  linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);		
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_side_link_kernel<1,0,1><<<halfGridDim, blockDim>>>( P3Odd,  P3Even, 
							       P3muOdd,  P3muEven,
							       TempxOdd,  TempxEven,
							       QmuOdd,   QmuEven,
							       shortPOdd,   shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);		

    }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	do_side_link_kernel<0,1,0><<<halfGridDim, blockDim>>>( P3Even,  P3Odd, 
							       P3muEven,  P3muOdd,
							       TempxEven,  TempxOdd,
							       QmuEven,   QmuOdd,
							       shortPEven,   shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven,  linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_side_link_kernel<0,1,1><<<halfGridDim, blockDim>>>( P3Odd,  P3Even, 
							       P3muOdd,  P3muEven,
							       TempxOdd,  TempxEven,
							       QmuOdd,   QmuEven,
							       shortPOdd,   shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);
	
    }else{
	do_side_link_kernel<0,0,0><<<halfGridDim, blockDim>>>( P3Even,  P3Odd, 
							       P3muEven,  P3muOdd,
							       TempxEven,  TempxOdd,
							       QmuEven,   QmuOdd,
							       shortPEven,   shortPOdd,
							       sig, mu, coeff, accumu_coeff,
							       linkEven, linkOdd,
							       momEven, momOdd,
							       momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);
	
	do_side_link_kernel<0,0,1><<<halfGridDim, blockDim>>>( P3Odd,  P3Even, 
							       P3muOdd,  P3muEven,
							       TempxOdd,  TempxEven,
							       QmuOdd,   QmuEven,
							       shortPOdd,   shortPEven,
							       sig, mu, coeff, accumu_coeff,
							       linkOdd, linkEven,
							       momOdd, momEven,
							       momMatrixOdd, momMatrixEven);
    }
    
    cudaUnbindTexture(siteLink0TexSingle);
    cudaUnbindTexture(siteLink1TexSingle);    

}

template<int sig_positive, int mu_positive, int oddBit>
__global__ void
do_all_link_kernel(float2* tempxxEven, 
		float2* QprevOdd,
		float2* PmuEven, float2* PmuOdd,
		float2* P3Even, float2* P3Odd,
		float2* P3muEven, float2* P3muOdd,
		float2* shortPEven, float2* shortPOdd,
		int sig, int mu, int lambda, int kappa, // mu is the current side direction, lambda is the last side direction, kappa is the side link direction before that
		float coeff, float mcoeff, float accumu_coeff,
		float4* linkEven, float4* linkOdd,
		float2* momEven, float2* momOdd,
		float2* momMatrixEven, float2* momMatrixOdd)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;

    int z1 = FAST_INT_DIVIDE(sid, X1h);
    int x1h = sid - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    int X = 2*sid + x1odd;
    
    int new_x1, new_x2, new_x3, new_x4;
    int ad_link_sign=1;
    int ab_link_sign=1;
    int bc_link_sign=1;   
    
    float4 LINKA0, LINKA1, LINKA2, LINKA3, LINKA4;
    float4 LINKB0, LINKB1, LINKB2, LINKB3, LINKB4;
    float4 LINKC0, LINKC1, LINKC2, LINKC3, LINKC4;
    float4 TEMP0,  TEMP1,  TEMP2,  TEMP3,  TEMP4;
    float2 LINKD0, LINKD1, LINKD2, LINKD3, LINKD4, LINKD5, LINKD6, LINKD7, LINKD8;
    float2 OPROD0, OPROD1, OPROD2, OPROD3, OPROD4, OPROD5, OPROD6, OPROD7, OPROD8;
    float2 LINKAB0, LINKAB1, LINKAB2, LINKAB3, LINKAB4, LINKAB5, LINKAB6, LINKAB7, LINKAB8;
    float2 AH0, AH1, AH2, AH3, AH4;
    float2 STORE0, STORE1, STORE2, STORE3, STORE4, STORE5, STORE6, STORE7, STORE8; 

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
    new_x1 = x1;
    new_x2 = x2;
    new_x3 = x3;
    new_x4 = x4;

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
	FF_COMPUTE_RECONSTRUCT_SIGN(ad_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }else{
	ad_link_nbr_idx = sid;
	FF_COMPUTE_RECONSTRUCT_SIGN(ad_link_sign, mymu, x1, x2, x3, x4);	
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
	FF_COMPUTE_RECONSTRUCT_SIGN(bc_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }
    
    new_x1 = x1;
    new_x2 = x2;
    new_x3 = x3;
    new_x4 = x4;
    if(sig_positive){
	FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
    }else{
	FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
    }
    point_b = (new_mem_idx >> 1);
    if (!mu_positive){
	bc_link_nbr_idx = point_b;
	FF_COMPUTE_RECONSTRUCT_SIGN(bc_link_sign, mymu, new_x1,new_x2,new_x3,new_x4);
    }      
    
    if(sig_positive){
	ab_link_nbr_idx = sid;
	FF_COMPUTE_RECONSTRUCT_SIGN(ab_link_sign, mysig, x1, x2, x3, x4);	
    }else{	
	ab_link_nbr_idx = point_b;
	FF_COMPUTE_RECONSTRUCT_SIGN(ab_link_sign, mysig, new_x1,new_x2,new_x3,new_x4);

    }
   // Code added by J.F.
    LOAD_MATRIX_18_SINGLE(QprevOdd, point_d, LINKAB);
    ASSIGN_MAT(linkab, linka);
  
    if (mu_positive){
	FF_LOAD_MATRIX(linkOdd, mymu, ad_link_nbr_idx, LINKC);
    }else{
	FF_LOAD_MATRIX(linkEven, mymu, ad_link_nbr_idx, LINKC);
    }
    RECONSTRUCT_LINK_12(mymu, ad_link_nbr_idx, ad_link_sign, linkc);

    if (mu_positive){
	MAT_MUL_MAT(linka, linkc, linkd);
    }else{
	MAT_MUL_ADJ_MAT(linka, linkc, linkd);
    }
 
    LOAD_MATRIX_18_SINGLE(tempxxEven, point_c, OPROD);

    if (mu_positive){
	FF_LOAD_MATRIX(linkEven, mymu, bc_link_nbr_idx, LINKA);
    }else{
	FF_LOAD_MATRIX(linkOdd, mymu, bc_link_nbr_idx, LINKA);	
    }
    RECONSTRUCT_LINK_12(mymu, bc_link_nbr_idx, bc_link_sign, linka);


    if (mu_positive){    
	ADJ_MAT_MUL_MAT(linka, oprod, linkb);
    }else{
	MAT_MUL_MAT(linka, oprod, linkb);
    }
    // linkb now connects site b to the outer product! 
    // Done with LINKA for the time being.	

    if (sig_positive){
	FF_LOAD_MATRIX(linkEven, mysig, ab_link_nbr_idx, LINKA);
    }else{
	FF_LOAD_MATRIX(linkOdd, mysig, ab_link_nbr_idx, LINKA);
    }

    RECONSTRUCT_LINK_12(mysig, ab_link_nbr_idx, ab_link_sign, linka);

    float mycoeff; // needed further down


    if (sig_positive){        
	MAT_MUL_MAT(linka, linkb, oprod);
    }else{
	ADJ_MAT_MUL_MAT(linka, linkb, oprod);
    }
    // oprod now connects site a to the outer product 
    // Force from the forward link in the staple
    if (sig_positive){	
	if(oddBit){mycoeff = coeff;}
	else{mycoeff = -coeff;}	

	MAT_MUL_MAT(linkb, linkd, temp);
	ASSIGN_MAT(temp, linkd);

 	LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);
        SCALAR_MULT_ADD_SU3_MATRIX(store, linkd, mycoeff, store);
        WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, STORE);

/*
	SCALAR_MULT_MATRIX(mycoeff, linkd, linkd);
	WRITE_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, LINKD);
        LOAD_MOM_MATRIX_SINGLE(momMatrixEven, sig, sid, LINKD); 
        MAT_MUL_MAT(linka, linkd, linkb);
	SIMPLE_MAT_FORCE_TO_MOM(linkb, momEven, sid, sig);
*/
    }




   // QprevOdd = linkab
   if (mu_positive){

     if(sig_positive){
       if(oddBit==1){mycoeff = coeff;}
       else{mycoeff = -coeff;}
     }else{
       if(oddBit==1){mycoeff = -coeff;}
       else{mycoeff = coeff;}	
     }	

     MAT_MUL_MAT(oprod, linkab, temp);

     LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, STORE);
     SCALAR_MULT_ADD_SU3_MATRIX(store, temp, mycoeff, store);
     WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, STORE);

    /*
     SCALAR_MULT_MATRIX(mycoeff, temp, linkab);
     WRITE_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, LINKAB);
     LOAD_MOM_MATRIX_SINGLE(momMatrixOdd, mu, point_d, LINKAB);
     MAT_MUL_MAT(linkc, linkab, linkd);	
     SIMPLE_MAT_FORCE_TO_MOM(linkd, momOdd, point_d, mu);
    */
     MAT_MUL_MAT(linkc, oprod, linkd);	
   }else{

     if(sig_positive){
       if(oddBit==1){mycoeff = coeff;}
       else{mycoeff = -coeff;}
     }else{
       if(oddBit==1){mycoeff = -coeff;}
       else{mycoeff = coeff;}
     }	

     ADJ_MAT_MUL_ADJ_MAT(linkab, oprod, temp);	

     LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, STORE);
     SCALAR_MULT_ADD_SU3_MATRIX(store, temp, mycoeff, store);
     WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, STORE);
    /*
     SCALAR_MULT_MATRIX(mycoeff, temp, linkab);
     WRITE_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, LINKAB);
     LOAD_MOM_MATRIX_SINGLE(momMatrixEven, OPP_DIR(mu), sid, LINKAB);
     MAT_MUL_MAT(linkc, linkab, linkb); // linkc = U[mu](x)
     SIMPLE_MAT_FORCE_TO_MOM(linkb, momEven, sid, OPP_DIR(mu));
   */
     ADJ_MAT_MUL_MAT(linkc, oprod, linkd);	
   }
		

   LOAD_MATRIX_18_SINGLE(shortPOdd, point_d, OPROD);
   SCALAR_MULT_ADD_MATRIX(oprod, linkd, accumu_coeff, oprod);
   WRITE_MATRIX_18_SINGLE(shortPOdd, point_d, OPROD);
}



static void
all_link_kernel(float2* tempxEven, float2* tempxOdd,
		float2* QprevEven, float2* QprevOdd, 
		float2* PmuEven, float2* PmuOdd,
		float2* P3Even, float2* P3Odd,
		float2* P3muEven, float2* P3muOdd,
		float2* shortPEven, float2* shortPOdd,
		int sig, int mu, int lambda, int kappa, 
		float coeff, float mcoeff, float accumu_coeff,
		float4* linkEven, float4* linkOdd, FullGauge cudaSiteLink,
		float2* momEven, float2* momOdd,
		dim3 gridDim, dim3 blockDim,
		float2* momMatrixEven, float2* momMatrixOdd)
		   
{
    dim3 halfGridDim(gridDim.x/2, 1,1);

    cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);
    cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
    
    if (GOES_FORWARDS(sig) && GOES_FORWARDS(mu)){		
	do_all_link_kernel<1,1,0><<<halfGridDim, blockDim>>>( tempxEven,  
							      QprevOdd, 
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd,
							      momMatrixEven, momMatrixOdd);
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);
	do_all_link_kernel<1,1,1><<<halfGridDim, blockDim>>>( tempxOdd,  
							      QprevEven,
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);	

	
    }else if (GOES_FORWARDS(sig) && GOES_BACKWARDS(mu)){

	do_all_link_kernel<1,0,0><<<halfGridDim, blockDim>>>( tempxEven,   
							      QprevOdd,
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd,
							      momMatrixEven, momMatrixOdd);	
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_all_link_kernel<1,0,1><<<halfGridDim, blockDim>>>( tempxOdd,  
							      QprevEven, 
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);	
	
    }else if (GOES_BACKWARDS(sig) && GOES_FORWARDS(mu)){
	do_all_link_kernel<0,1,0><<<halfGridDim, blockDim>>>( tempxEven,  
							      QprevOdd, 
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd, 
							      momMatrixEven, momMatrixOdd);	
	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	
	do_all_link_kernel<0,1,1><<<halfGridDim, blockDim>>>( tempxOdd,  
							      QprevEven, 
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, lambda, kappa, 
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);		
    }else{
	do_all_link_kernel<0,0,0><<<halfGridDim, blockDim>>>( tempxEven, 
							      QprevOdd, 
							      PmuEven,  PmuOdd,
							      P3Even,  P3Odd,
							      P3muEven,  P3muOdd,
							      shortPEven,  shortPOdd,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkEven, linkOdd,
							      momEven, momOdd,
							      momMatrixEven, momMatrixOdd);	

	cudaUnbindTexture(siteLink0TexSingle);
	cudaUnbindTexture(siteLink1TexSingle);

	//opposive binding
	cudaBindTexture(0, siteLink0TexSingle, cudaSiteLink.odd, cudaSiteLink.bytes);
	cudaBindTexture(0, siteLink1TexSingle, cudaSiteLink.even, cudaSiteLink.bytes);

	do_all_link_kernel<0,0,1><<<halfGridDim, blockDim>>>( tempxOdd,  
							      QprevEven, 
							      PmuOdd,  PmuEven,
							      P3Odd,  P3Even,
							      P3muOdd,  P3muEven,
							      shortPOdd,  shortPEven,
							      sig,  mu, lambda, kappa,
							      coeff, mcoeff, accumu_coeff,
							      linkOdd, linkEven,
							      momOdd, momEven,
							      momMatrixOdd, momMatrixEven);	
    }

    cudaUnbindTexture(siteLink0TexSingle);
    cudaUnbindTexture(siteLink1TexSingle);
}


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
    float4 LINKA0, LINKA1, LINKA2, LINKA3, LINKA4;
    float4 LINKB0, LINKB1, LINKB2, LINKB3, LINKB4;
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
    
    int z1 = FAST_INT_DIVIDE(sid, X1h);
    int x1h = sid - z1*X1h;
    int z2 = FAST_INT_DIVIDE(z1, X2);
    int x2 = z1 - z2*X2;
    int x4 = FAST_INT_DIVIDE(z2, X3);
    int x3 = z2 - x4*X3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;
    //int X = 2*sid + x1odd;
    
    int dx[4];
    int new_x1, new_x2, new_x3, new_x4, new_idx;
    int sign=1;
    
    if (GOES_BACKWARDS(mu)){
	//The one link
	LOAD_HW(myPmu, sid, HWA);
	LOAD_HW(myTempx, sid, HWB);
	ADD_FORCE_TO_MOM(hwa, hwb, myMom, sid, OPP_DIR(mu), OneLink, oddBit);
	
	//Naik term
	dx[3]=dx[2]=dx[1]=dx[0]=0;
	dx[OPP_DIR(mu)] = -1;
	new_x1 = (x1 + dx[0] + X1)%X1;
	new_x2 = (x2 + dx[1] + X2)%X2;
	new_x3 = (x3 + dx[2] + X3)%X3;
	new_x4 = (x4 + dx[3] + X4)%X4;	
	new_idx = (new_x4*X3X2X1+new_x3*X2X1+new_x2*X1+new_x1) >> 1;
	LOAD_HW(otherTempx, new_idx, HWA);
	LOAD_MATRIX(otherLink, OPP_DIR(mu), new_idx, LINKA);
	FF_COMPUTE_RECONSTRUCT_SIGN(sign, OPP_DIR(mu), new_x1,new_x2,new_x3,new_x4);
	RECONSTRUCT_LINK_12(OPP_DIR(mu), new_idx, sign, linka);		
	ADJ_MAT_MUL_HW(linka, hwa, hwc); //Popmu
	
	LOAD_HW(myPnumu, sid, HWD);
	ADD_FORCE_TO_MOM(hwd, hwc, myMom, sid, OPP_DIR(mu), mNaik, oddBit);
	
	dx[3]=dx[2]=dx[1]=dx[0]=0;
	dx[OPP_DIR(mu)] = 1;
	new_x1 = (x1 + dx[0] + X1)%X1;
	new_x2 = (x2 + dx[1] + X2)%X2;
	new_x3 = (x3 + dx[2] + X3)%X3;
	new_x4 = (x4 + dx[3] + X4)%X4;	
	new_idx = (new_x4*X3X2X1+new_x3*X2X1+new_x2*X1+new_x1) >> 1;
	LOAD_HW(otherPnumu, new_idx, HWA);
	LOAD_MATRIX(myLink, OPP_DIR(mu), sid, LINKA);
	FF_COMPUTE_RECONSTRUCT_SIGN(sign, OPP_DIR(mu), x1, x2, x3, x4);
	RECONSTRUCT_LINK_12(OPP_DIR(mu), sid, sign, linka);	
	MAT_MUL_HW(linka, hwa, hwc);
	ADD_FORCE_TO_MOM(hwc, hwb, myMom, sid, OPP_DIR(mu), Naik, oddBit);	
    }else{
	dx[3]=dx[2]=dx[1]=dx[0]=0;
	dx[mu] = 1;
	new_x1 = (x1 + dx[0] + X1)%X1;
	new_x2 = (x2 + dx[1] + X2)%X2;
	new_x3 = (x3 + dx[2] + X3)%X3;
	new_x4 = (x4 + dx[3] + X4)%X4;	
	new_idx = (new_x4*X3X2X1+new_x3*X2X1+new_x2*X1+new_x1) >> 1;
	LOAD_HW(otherTempx, new_idx, HWA);
	LOAD_MATRIX(myLink, mu, sid, LINKA);
	FF_COMPUTE_RECONSTRUCT_SIGN(sign, mu, x1, x2, x3, x4);
	RECONSTRUCT_LINK_12(mu, sid, sign, linka);
	MAT_MUL_HW(linka, hwa, hwb);
	
	LOAD_HW(myPnumu, sid, HWC);
	ADD_FORCE_TO_MOM(hwb, hwc, myMom, sid, mu, Naik, oddBit);
	

    }
}

/*
#define Pmu          tempvec[0] 
#define Pnumu        tempvec[1]
#define Prhonumu     tempvec[2]
#define P7           tempvec[3]
#define P7rho        tempvec[4]              
#define P7rhonu      tempvec[5]
#define P5           tempvec[6]
#define P3           tempvec[7]
#define P5nu         tempvec[3]
#define P3mu         tempvec[3]
#define Popmu        tempvec[4]
#define Pmumumu      tempvec[4]
*/
#define Pmu 	  tempmat[0]
#define Pnumu     tempmat[1]
#define Prhonumu  tempmat[2]
#define P7 	  tempmat[3]
#define P7rho     tempmat[4]
//#define P7rhonu   tempmat[5] // never used
#define P5	  tempmat[5]
#define P3        tempmat[6]
#define P5nu	  tempmat[3]
#define P3mu	  tempmat[3]
#define Popmu	  tempmat[4]
#define Pmumumu	  tempmat[4]

// Here, we have a problem. 
// If we use float2 to store the Ps and float2 
// for the Qs. We can't use the same temporaries 
// here. 
// I wonder which is better? 
// To use float2 for both Ps and Qs and 
// and use the same temporary matrices 
// or use float2 for the Ps and float4 for the 
// Qs and use separate sets of matrices?
// Ultimately, I will use float4 for the Q matrices 
// for the first level of smearing and float2 
// for the Q matrices for the second level of smearing. 
// To begin with, use float4 for everything.
// Note, I will have to go back and undo the float4s above.


// if first level of smearing
// 
 #define Qmu      tempCmat[0]
 #define Qnumu    tempCmat[1]
 #define Qrhonumu tempCmat[2] 
 #define Q3       tempCmat[3] // not sure if I really need this
 #define Null	  tempCmat[3] // not used
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
    float2 coeff;
    
    float2 OneLink, Lepage, Naik, FiveSt, ThreeSt, SevenSt;
    float2 mNaik, mLepage, mFiveSt, mThreeSt, mSevenSt;
    
    Real ferm_epsilon;
    ferm_epsilon = 2.0*weight1*eps;
    OneLink.x = act_path_coeff[0]*ferm_epsilon ;
    Naik.x    = act_path_coeff[1]*ferm_epsilon ; mNaik.x    = -Naik.x;
    ThreeSt.x = act_path_coeff[2]*ferm_epsilon ; mThreeSt.x = -ThreeSt.x;
    FiveSt.x  = act_path_coeff[3]*ferm_epsilon ; mFiveSt.x  = -FiveSt.x;
    SevenSt.x = act_path_coeff[4]*ferm_epsilon ; mSevenSt.x = -SevenSt.x;
    Lepage.x  = act_path_coeff[5]*ferm_epsilon ; mLepage.x  = -Lepage.x;
    
    ferm_epsilon = 2.0*weight2*eps;
    OneLink.y = act_path_coeff[0]*ferm_epsilon ;
    Naik.y    = act_path_coeff[1]*ferm_epsilon ; mNaik.y    = -Naik.y;
    ThreeSt.y = act_path_coeff[2]*ferm_epsilon ; mThreeSt.y = -ThreeSt.y;
    FiveSt.y  = act_path_coeff[3]*ferm_epsilon ; mFiveSt.y  = -FiveSt.y;
    SevenSt.y = act_path_coeff[4]*ferm_epsilon ; mSevenSt.y = -SevenSt.y;
    Lepage.y  = act_path_coeff[5]*ferm_epsilon ; mLepage.y  = -Lepage.y;
    
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
				(float2*)Null.even.data,(float2*)Null.odd.data,
				(float2*)Qmu.even.data, (float2*)Qmu.odd.data,
			        (float2*)Q3.even.data, (float2*)Q3.odd.data,
				//sig, mu, mThreeSt.x, // --> sig, mu, null, mThreeSt.x, where null is defined to be -1 above and is therefore not a valid direction
				sig, mu, null, mThreeSt.x,
				(float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
				(float2*)cudaMom.even, (float2*)cudaMom.odd, 
				gridDim, blockDim, true,
			        (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); // I have added true and false to indicate 
							  // whether I am on a three staple




	
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
				    (float2*)Q5.even.data, (float2*)Q5.odd.data,
				    //sig, nu, FiveSt.x, // --> sig, nu, mu, FiveSt.x
				    sig, nu, mu, FiveSt.x,
				    (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
				    (float2*)cudaMom.even, (float2*)cudaMom.odd,
				    gridDim, blockDim, false,
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
		    if(FiveSt.x != 0)coeff.x = SevenSt.x/FiveSt.x ; else coeff.x = 0;
		    if(FiveSt.y != 0)coeff.y = SevenSt.y/FiveSt.y ; else coeff.y = 0;		    
		    all_link_kernel((float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
				    (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				    (float2*)Prhonumu.even.data, (float2*)Prhonumu.odd.data,
				    (float2*)P7.even.data, (float2*)P7.odd.data,
				    (float2*)P7rho.even.data, (float2*)P7rho.odd.data,
				    (float2*)P5.even.data, (float2*)P5.odd.data,
				    sig, rho, nu, mu, SevenSt.x,mSevenSt.x,coeff.x,
				    (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
				    (float2*)cudaMom.even, (float2*)cudaMom.odd,
				    gridDim, blockDim,
				    (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);	
		    checkCudaError();

		}//rho  		

		//5-link: side link
		//kernel B2
		if(ThreeSt.x != 0)coeff.x = FiveSt.x/ThreeSt.x ; else coeff.x = 0;
		if(ThreeSt.y != 0)coeff.y = FiveSt.y/ThreeSt.y ; else coeff.y = 0;
		side_link_kernel((float2*)P5.even.data, (float2*)P5.odd.data,
				 (float2*)P5nu.even.data, (float2*)P5nu.odd.data,
				 (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
				 (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				 (float2*)P3.even.data, (float2*)P3.odd.data,
				 sig, nu, mFiveSt.x, coeff.x,
				 (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
				 (float2*)cudaMom.even, (float2*)cudaMom.odd,
				 gridDim, blockDim,
				 (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
		checkCudaError();



	    }//nu

	    //lepage
	    //Kernel A2
	    middle_link_kernel( (float2*)Pmu.even.data, (float2*)Pmu.odd.data,
				(float2*)Pnumu.even.data, (float2*)Pnumu.odd.data,
				(float2*)P5.even.data, (float2*)P5.odd.data,
				(float2*)Qmu.even.data, (float2*)Qmu.odd.data, // input Q matrix
				(float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
				(float2*)Q5.even.data, (float2*)Q5.odd.data,
				sig, mu, mu, Lepage.x,
				(float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink, 
				(float2*)cudaMom.even, (float2*)cudaMom.odd,
				gridDim, blockDim, false,
				(float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd); // not on a threeStaple => have to read in Qprev   
	    checkCudaError();		
	    
	    if(ThreeSt.x != 0)coeff.x = Lepage.x/ThreeSt.x ; else coeff.x = 0;
	    if(ThreeSt.y != 0)coeff.y = Lepage.y/ThreeSt.y ; else coeff.y = 0;
	    
	    side_link_kernel((float2*)P5.even.data, (float2*)P5.odd.data,
			     (float2*)P5nu.even.data, (float2*)P5nu.odd.data,
			     (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
			     (float2*)Qnumu.even.data, (float2*)Qnumu.odd.data,
			     (float2*)P3.even.data, (float2*)P3.odd.data,
			     sig, mu, mLepage.x ,coeff.x,
			     (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
			     (float2*)cudaMom.even, (float2*)cudaMom.odd,
			     gridDim, blockDim,
			     (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
	    checkCudaError();		


	    //3-link side link
	    coeff.x=coeff.y=0;

	    side_link_kernel((float2*)P3.even.data, (float2*)P3.odd.data,
			     (float2*)P3mu.even.data, (float2*)P3mu.odd.data,
			     (float2*)NULL, (float2*)NULL,
			     (float2*)Qmu.even.data, (float2*)Qmu.odd.data,
			     (float2*)NULL, (float2*)NULL,
			     sig, mu, ThreeSt.x, coeff.x,
			     (float4*)cudaSiteLink.even, (float4*)cudaSiteLink.odd, cudaSiteLink,
			     (float2*)cudaMom.even, (float2*)cudaMom.odd,
			     gridDim, blockDim,
			     (float2*)cudaMomMatrix.even, (float2*)cudaMomMatrix.odd);
	    checkCudaError();			    




/*
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
*/    
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

#undef Popmu
#undef Pmumumu

#undef Qmu
#undef Qnumu
#undef Qrhonumu
#undef Q3
/*
void hisq_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
		   float2* cudaOprodEven, float2* cudaOprodOdd, FullGauge cudaSiteLink, FullMom cudaMom, QudaGaugeParam* param)
*/
void
hisq_force_cuda(double eps, double weight1, double weight2, void* act_path_coeff,
		   FullOprod cudaOprod, FullGauge cudaSiteLink, FullMom cudaMom, FullGauge cudaMomMatrix, QudaGaugeParam* param)
{

    FullMatrix tempmat[7];
    for(int i=0; i<7; i++){
	tempmat[i]  = createMatQuda(param->X, param->cuda_prec);
    }

    FullMatrix tempCompmat[4];
    for(int i=0; i<4; i++){
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
