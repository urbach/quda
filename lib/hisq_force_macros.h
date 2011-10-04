#ifndef _HISQ_FORCE_MACROS_H_
#define _HISQ_FORCE_MACROS_H_

/*
#define HISQ_SINGLE_PRECISION
// #define DOUBLE_PRECISION

#ifdef(HISQ_SINGLE_PRECISION)

#endif
*/

#define link_W00_re LINK_W[0].x
#define link_W00_im LINK_W[0].y
#define link_W01_re LINK_W[0].z
#define link_W01_im LINK_W[0].w
#define link_W02_re LINK_W[1].x
#define link_W02_im LINK_W[1].y
#define link_W10_re LINK_W[1].z
#define link_W10_im LINK_W[1].w
#define link_W11_re LINK_W[2].x
#define link_W11_im LINK_W[2].y
#define link_W12_re LINK_W[2].z
#define link_W12_im LINK_W[2].w
#define link_W20_re LINK_W[3].x
#define link_W20_im LINK_W[3].y
#define link_W21_re LINK_W[3].z
#define link_W21_im LINK_W[3].w
#define link_W22_re LINK_W[4].x
#define link_W22_im LINK_W[4].y


#define link_X00_re LINK_X[0].x
#define link_X00_im LINK_X[0].y
#define link_X01_re LINK_X[0].z
#define link_X01_im LINK_X[0].w
#define link_X02_re LINK_X[1].x
#define link_X02_im LINK_X[1].y
#define link_X10_re LINK_X[1].z
#define link_X10_im LINK_X[1].w
#define link_X11_re LINK_X[2].x
#define link_X11_im LINK_X[2].y
#define link_X12_re LINK_X[2].z
#define link_X12_im LINK_X[2].w
#define link_X20_re LINK_X[3].x
#define link_X20_im LINK_X[3].y
#define link_X21_re LINK_X[3].z
#define link_X21_im LINK_X[3].w
#define link_X22_re LINK_X[4].x
#define link_X22_im LINK_X[4].y


#define link_Y00_re LINK_Y[0].x
#define link_Y00_im LINK_Y[0].y
#define link_Y01_re LINK_Y[0].z
#define link_Y01_im LINK_Y[0].w
#define link_Y02_re LINK_Y[1].x
#define link_Y02_im LINK_Y[1].y
#define link_Y10_re LINK_Y[1].z
#define link_Y10_im LINK_Y[1].w
#define link_Y11_re LINK_Y[2].x
#define link_Y11_im LINK_Y[2].y
#define link_Y12_re LINK_Y[2].z
#define link_Y12_im LINK_Y[2].w
#define link_Y20_re LINK_Y[3].x
#define link_Y20_im LINK_Y[3].y
#define link_Y21_re LINK_Y[3].z
#define link_Y21_im LINK_Y[3].w
#define link_Y22_re LINK_Y[4].x
#define link_Y22_im LINK_Y[4].y


#define link_Z00_re LINK_Z[0].x
#define link_Z00_im LINK_Z[0].y
#define link_Z01_re LINK_Z[0].z
#define link_Z01_im LINK_Z[0].w
#define link_Z02_re LINK_Z[1].x
#define link_Z02_im LINK_Z[1].y
#define link_Z10_re LINK_Z[1].z
#define link_Z10_im LINK_Z[1].w
#define link_Z11_re LINK_Z[2].x
#define link_Z11_im LINK_Z[2].y
#define link_Z12_re LINK_Z[2].z
#define link_Z12_im LINK_Z[2].w
#define link_Z20_re LINK_Z[3].x
#define link_Z20_im LINK_Z[3].y
#define link_Z21_re LINK_Z[3].z
#define link_Z21_im LINK_Z[3].w
#define link_Z22_re LINK_Z[4].x
#define link_Z22_im LINK_Z[4].y



// Color matrices stored as an array of float2 or 
// double 2
#define color_mat_W00_re COLOR_MAT_W[0].x
#define color_mat_W00_im COLOR_MAT_W[0].y
#define color_mat_W01_re COLOR_MAT_W[1].x
#define color_mat_W01_im COLOR_MAT_W[1].y
#define color_mat_W02_re COLOR_MAT_W[2].x
#define color_mat_W02_im COLOR_MAT_W[2].y
#define color_mat_W10_re COLOR_MAT_W[3].x
#define color_mat_W10_im COLOR_MAT_W[3].y 
#define color_mat_W11_re COLOR_MAT_W[4].x
#define color_mat_W11_im COLOR_MAT_W[4].y
#define color_mat_W12_re COLOR_MAT_W[5].x
#define color_mat_W12_im COLOR_MAT_W[5].y
#define color_mat_W20_re COLOR_MAT_W[6].x
#define color_mat_W20_im COLOR_MAT_W[6].y
#define color_mat_W21_re COLOR_MAT_W[7].x
#define color_mat_W21_im COLOR_MAT_W[7].y
#define color_mat_W22_re COLOR_MAT_W[8].x
#define color_mat_W22_im COLOR_MAT_W[8].y


#define color_mat_X00_re COLOR_MAT_X[0].x
#define color_mat_X00_im COLOR_MAT_X[0].y
#define color_mat_X01_re COLOR_MAT_X[1].x
#define color_mat_X01_im COLOR_MAT_X[1].y
#define color_mat_X02_re COLOR_MAT_X[2].x
#define color_mat_X02_im COLOR_MAT_X[2].y
#define color_mat_X10_re COLOR_MAT_X[3].x
#define color_mat_X10_im COLOR_MAT_X[3].y 
#define color_mat_X11_re COLOR_MAT_X[4].x
#define color_mat_X11_im COLOR_MAT_X[4].y
#define color_mat_X12_re COLOR_MAT_X[5].x
#define color_mat_X12_im COLOR_MAT_X[5].y
#define color_mat_X20_re COLOR_MAT_X[6].x
#define color_mat_X20_im COLOR_MAT_X[6].y
#define color_mat_X21_re COLOR_MAT_X[7].x
#define color_mat_X21_im COLOR_MAT_X[7].y
#define color_mat_X22_re COLOR_MAT_X[8].x
#define color_mat_X22_im COLOR_MAT_X[8].y


#define color_mat_Y00_re COLOR_MAT_Y[0].x
#define color_mat_Y00_im COLOR_MAT_Y[0].y
#define color_mat_Y01_re COLOR_MAT_Y[1].x
#define color_mat_Y01_im COLOR_MAT_Y[1].y
#define color_mat_Y02_re COLOR_MAT_Y[2].x
#define color_mat_Y02_im COLOR_MAT_Y[2].y
#define color_mat_Y10_re COLOR_MAT_Y[3].x
#define color_mat_Y10_im COLOR_MAT_Y[3].y 
#define color_mat_Y11_re COLOR_MAT_Y[4].x
#define color_mat_Y11_im COLOR_MAT_Y[4].y
#define color_mat_Y12_re COLOR_MAT_Y[5].x
#define color_mat_Y12_im COLOR_MAT_Y[5].y
#define color_mat_Y20_re COLOR_MAT_Y[6].x
#define color_mat_Y20_im COLOR_MAT_Y[6].y
#define color_mat_Y21_re COLOR_MAT_Y[7].x
#define color_mat_Y21_im COLOR_MAT_Y[7].y
#define color_mat_Y22_re COLOR_MAT_Y[8].x
#define color_mat_Y22_im COLOR_MAT_Y[8].y


#define color_mat_Z00_re COLOR_MAT_Z[0].x
#define color_mat_Z00_im COLOR_MAT_Z[0].y
#define color_mat_Z01_re COLOR_MAT_Z[1].x
#define color_mat_Z01_im COLOR_MAT_Z[1].y
#define color_mat_Z02_re COLOR_MAT_Z[2].x
#define color_mat_Z02_im COLOR_MAT_Z[2].y
#define color_mat_Z10_re COLOR_MAT_Z[3].x
#define color_mat_Z10_im COLOR_MAT_Z[3].y 
#define color_mat_Z11_re COLOR_MAT_Z[4].x
#define color_mat_Z11_im COLOR_MAT_Z[4].y
#define color_mat_Z12_re COLOR_MAT_Z[5].x
#define color_mat_Z12_im COLOR_MAT_Z[5].y
#define color_mat_Z20_re COLOR_MAT_Z[6].x
#define color_mat_Z20_im COLOR_MAT_Z[6].y
#define color_mat_Z21_re COLOR_MAT_Z[7].x
#define color_mat_Z21_im COLOR_MAT_Z[7].y
#define color_mat_Z22_re COLOR_MAT_Z[8].x
#define color_mat_Z22_im COLOR_MAT_Z[8].y



// Maybe I should rename it to 
// be COLOR_FIELD_LOAD_18_SINGLE
#define LOAD_MATRIX_18_SINGLE(gauge, idx, var)                  \
  var[0] = gauge[idx];                                          \
  var[1] = gauge[idx + Vh];                                     \
  var[2] = gauge[idx + Vhx2];                                   \
  var[3] = gauge[idx + Vhx3];                                   \
  var[4] = gauge[idx + Vhx4];                                   \
  var[5] = gauge[idx + Vhx5];                                   \
  var[6] = gauge[idx + Vhx6];                                   \
  var[7] = gauge[idx + Vhx7];                                   \
  var[8] = gauge[idx + Vhx8];

// COLOR_FIELD_WRITE_18_SINGLE
#define WRITE_MATRIX_18_SINGLE(mat, idx, var) do{ \
    mat[idx + 0*Vh] = var[0];  \
    mat[idx + 1*Vh] = var[1];  \
    mat[idx + 2*Vh] = var[2];  \
    mat[idx + 3*Vh] = var[3];  \
    mat[idx + 4*Vh] = var[4];  \
    mat[idx + 5*Vh] = var[5];  \
    mat[idx + 6*Vh] = var[6];  \
    mat[idx + 7*Vh] = var[7];  \
    mat[idx + 8*Vh] = var[8];  \
}while(0)



// GAUGE_FIELD_LOAD_18_SINGLE
#define LOAD_MOM_MATRIX_SINGLE(mom, dir, idx, var)      \
  var[0] = mom[idx + dir*Vhx9];                         \
  var[1] = mom[idx + dir*Vhx9 + Vh];                    \
  var[2] = mom[idx + dir*Vhx9 + Vhx2];                  \
  var[3] = mom[idx + dir*Vhx9 + Vhx3];                  \
  var[4] = mom[idx + dir*Vhx9 + Vhx4];                  \
  var[5] = mom[idx + dir*Vhx9 + Vhx5];                  \
  var[6] = mom[idx + dir*Vhx9 + Vhx6];                  \
  var[7] = mom[idx + dir*Vhx9 + Vhx7];                  \
  var[8] = mom[idx + dir*Vhx9 + Vhx8];



// GAUGE_FIELD_WRITE_18_SINGLE
#define WRITE_MOM_MATRIX_SINGLE(mat, dir, idx, var) do{ \
    mat[idx + dir*Vhx9 + 0*Vh] = var[0];  \
    mat[idx + dir*Vhx9 + 1*Vh] = var[1];  \
    mat[idx + dir*Vhx9 + 2*Vh] = var[2];  \
    mat[idx + dir*Vhx9 + 3*Vh] = var[3];  \
    mat[idx + dir*Vhx9 + 4*Vh] = var[4];  \
    mat[idx + dir*Vhx9 + 5*Vh] = var[5];  \
    mat[idx + dir*Vhx9 + 6*Vh] = var[6];  \
    mat[idx + dir*Vhx9 + 7*Vh] = var[7];  \
    mat[idx + dir*Vhx9 + 8*Vh] = var[8];  \
}while(0)




// matrix macros:
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


#define SU3_PROJECTOR(va, vb, m)                               \
   m##00_re = va##0_re * vb##0_re + va##0_im * vb##0_im;       \
  m##00_im = va##0_im * vb##0_re - va##0_re * vb##0_im;         \
  m##01_re = va##0_re * vb##1_re + va##0_im * vb##1_im;         \
  m##01_im = va##0_im * vb##1_re - va##0_re * vb##1_im;         \
  m##02_re = va##0_re * vb##2_re + va##0_im * vb##2_im;         \
  m##02_im = va##0_im * vb##2_re - va##0_re * vb##2_im;         \
  m##10_re = va##1_re * vb##0_re + va##1_im * vb##0_im;         \
  m##10_im = va##1_im * vb##0_re - va##1_re * vb##0_im;         \
  m##11_re = va##1_re * vb##1_re + va##1_im * vb##1_im;         \
  m##11_im = va##1_im * vb##1_re - va##1_re * vb##1_im;         \
  m##12_re = va##1_re * vb##2_re + va##1_im * vb##2_im;         \
  m##12_im = va##1_im * vb##2_re - va##1_re * vb##2_im;         \
  m##20_re = va##2_re * vb##0_re + va##2_im * vb##0_im;         \
  m##20_im = va##2_im * vb##0_re - va##2_re * vb##0_im;         \
  m##21_re = va##2_re * vb##1_re + va##2_im * vb##1_im;         \
  m##21_im = va##2_im * vb##1_re - va##2_re * vb##1_im;         \
  m##22_re = va##2_re * vb##2_re + va##2_im * vb##2_im;         \
  m##22_im = va##2_im * vb##2_re - va##2_re * vb##2_im;

// vc = va + vb*s 
#define SCALAR_MULT_ADD_SU3_VECTOR(va, vb, s, vc) do {  \
    vc##0_re = va##0_re + vb##0_re * s;         \
    vc##0_im = va##0_im + vb##0_im * s;         \
    vc##1_re = va##1_re + vb##1_re * s;         \
    vc##1_im = va##1_im + vb##1_im * s;         \
    vc##2_re = va##2_re + vb##2_re * s;         \
    vc##2_im = va##2_im + vb##2_im * s;         \
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










#endif // _HISQ_FORCE_MACROS_H_
