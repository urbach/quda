// dslash_def.h - Dslash kernel definitions

// There are currently 64 different variants of the Dslash kernel,
// each one characterized by a set of 6 options, where each option can
// take one of two values (2^6 = 64).  This file is structured so that
// the C preprocessor loops through all 64 variants (in a manner
// resembling a binary counter), sets the appropriate macros, and
// defines the corresponding functions.
//
// As an example of the function naming conventions, consider
//
// dslashSHS12DaggerXpayKernel(float4* g_out, int oddBit, float a).
//
// This is a Dslash^dagger kernel where the gauge field is read in single
// precision (S), the spinor field is read in half precision (H), the clover
// term is read in single precision (S), each gauge matrix is reconstructed
// from 12 real numbers, and the result is multiplied by "a" and summed
// with an input vector (Xpay).  More generally, each function name is given
// by the concatenation of the following 6 fields, with "dslash" at the
// beginning and "Kernel" at the end:
//
// DD_GPREC_F = D, S, H
// DD_SPREC_F = D, S, H
// DD_CPREC_F = D, S, H, [blank]; the latter corresponds to plain Wilson
// DD_RECON_F = 12, 8, 18
// DD_DAG_F = Dagger, [blank]
// DD_XPAY_F = Xpay, [blank], Axpy

// initialize on first iteration

#ifndef DD_LOOP
#define DD_LOOP
#define DD_DAG 0
#define DD_XPAY 0
#define DD_RECON 0
#define DD_GPREC 0
#define DD_SPREC 0
#define DD_CPREC 3
#endif

// set options for current iteration

#if (DD_DAG==0) // no dagger
#define DD_DAG_F
#else           // dagger
#define DD_DAG_F Dagger
#endif

#if (DD_XPAY==0) // no xpay 
#define DD_XPAY_F 
#define DD_PARAM2 int oddBit
#else            // xpay
#if (DD_XPAY==1)
#define DD_XPAY_F Xpay
#define DSLASH_XPAY
#else
#define DD_XPAY_F Axpy
#define DSLASH_AXPY
#endif
#if (DD_SPREC == 0)
#define DD_PARAM2 int oddBit, double a
#else
#define DD_PARAM2 int oddBit, float a
#endif
#endif



#if (DD_GPREC==0) // double-precision gauge field
#define DD_GPREC_F D
#define FATLINK0TEX fatLink0TexDouble
#define FATLINK1TEX fatLink1TexDouble
#define LONGLINK0TEX longLink0TexDouble
#define LONGLINK1TEX longLink1TexDouble
#elif (DD_GPREC==1) // single-precision gauge field
#define DD_GPREC_F S
#define FATLINK0TEX fatLink0TexSingle
#define FATLINK1TEX fatLink1TexSingle
#define LONGLINK0TEX longLink0TexSingle
#define LONGLINK1TEX longLink1TexSingle

#else             // half-precision gauge field
#define DD_GPREC_F H
#define FATLINK0TEX fatLink0TexHalf
#define FATLINK1TEX fatLink1TexHalf
#define LONGLINK0TEX longLink0TexHalf
#define LONGLINK1TEX longLink1TexHalf
#endif

#if (DD_RECON==0) // reconstruct from 12 reals
#define DD_RECON_F 12
#if (DD_GPREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_DOUBLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_DOUBLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_DOUBLE
#elif (DD_GPREC==1)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_SINGLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_SINGLE
#else
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_HALF
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_HALF
#endif
#elif (DD_RECON==1)            // reconstruct from 8 reals

#define DD_RECON_F 8
#if (DD_GPREC==0)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_DOUBLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_DOUBLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_8_DOUBLE
#elif (DD_GPREC==1)
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_SINGLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_8_SINGLE
#else
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_8_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_HALF
#define READ_LONG_MATRIX READ_LONG_MATRIX_8_HALF
#endif

#else //DD_RECON 18 reconstruct

#define DD_RECON_F 18
#if (DD_GPREC==0) //not supported
#define RECONSTRUCT_GAUGE_MATRIX(dir, gauge, idx, sign)
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_DOUBLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_18_DOUBLE
#elif (DD_GPREC==1)
#define RECONSTRUCT_GAUGE_MATRIX(dir, gauge, idx, sign) 
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_SINGLE
#define READ_LONG_MATRIX READ_LONG_MATRIX_18_SINGLE
#undef LONGLINK0TEX 
#undef LONGLINK1TEX 
#define LONGLINK0TEX longLink0TexSingle_norecon
#define LONGLINK1TEX longLink1TexSingle_norecon
#else //not supported
#define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_GAUGE_MATRIX_12_SINGLE
#define READ_FAT_MATRIX READ_FAT_MATRIX_18_HALF
#define READ_LONG_MATRIX READ_LONG_MATRIX_12_HALF
#endif
#endif


#if (DD_SPREC==0) // double-precision spinor field
#define DD_SPREC_F D
#define DD_PARAM1 double2* g_out
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX spinorTexDouble
#define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_DOUBLE
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_DOUBLE
#if (DD_XPAY== 1|| DD_XPAY == 2)
#define ACCUMTEX accumTexDouble
#define READ_ACCUM READ_ACCUM_DOUBLE
#endif
#elif (DD_SPREC==1) // single-precision spinor field
#define DD_SPREC_F S
#define DD_PARAM1 float2* g_out
 #define READ_SPINOR READ_SPINOR_SINGLE
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_SINGLE
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX spinorTexSingle
//#define WRITE_SPINOR WRITE_SPINOR_FLOAT4
#define WRITE_SPINOR WRITE_SPINOR_FLOAT2
#if (DD_XPAY==1 || DD_XPAY == 2)
#define ACCUMTEX accumTexSingle
#define READ_ACCUM READ_ACCUM_SINGLE
#endif
#else            // half-precision spinor field
#define DD_SPREC_F H
#define READ_SPINOR READ_SPINOR_HALF
#define READ_1ST_NBR_SPINOR READ_1ST_NBR_SPINOR_HALF
#define READ_3RD_NBR_SPINOR READ_3RD_NBR_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX spinorTexHalf
#define DD_PARAM1 short2* g_out, float *c
//#define WRITE_SPINOR WRITE_SPINOR_SHORT4
#define WRITE_SPINOR WRITE_SPINOR_SHORT2
#if (DD_XPAY==1 || DD_XPAY== 2)
#define ACCUMTEX accumTexHalf
#define READ_ACCUM READ_ACCUM_HALF
#endif
#endif


#define DD_CPREC_F

#define DD_CONCAT(g,s,c,r,d,x) dslash ## g ## s ## c ## r ## d ## x ## Kernel
#define DD_FUNC(g,s,c,r,d,x) DD_CONCAT(g,s,c,r,d,x)

// define the kernel
//#if (DD_RECON ==2)
//#if (DD_GPREC ==1 && DD_SPREC==1)
//#if (DD_SPREC==1 && DD_GPREC == 1 && DD_RECON == 1 &&DD_CPREC == 3) ||(DD_RECON ==2  && DD_GPREC == 1) 
#if 0
__global__ void
DD_FUNC(DD_GPREC_F, DD_SPREC_F, DD_CPREC_F, DD_RECON_F, DD_DAG_F, DD_XPAY_F)(DD_PARAM1, DD_PARAM2) {
#define SHARED_FLOATS_PER_THREAD 0
#include "dslash_core.h"
    
}

#else

#define SHARED_FLOATS_PER_THREAD 0
__global__ void
DD_FUNC(DD_GPREC_F, DD_SPREC_F, DD_CPREC_F, DD_RECON_F, DD_DAG_F, DD_XPAY_F)(DD_PARAM1, DD_PARAM2) {

}

#endif


// clean up

#undef DD_GPREC_F
#undef DD_SPREC_F
#undef DD_CPREC_F
#undef DD_RECON_F
#undef DD_DAG_F
#undef DD_XPAY_F
#undef DD_PARAM1
#undef DD_PARAM2
#undef DD_CONCAT
#undef DD_FUNC

#undef DSLASH_XPAY
#undef DSLASH_AXPY
#undef RECONSTRUCT_GAUGE_MATRIX
#undef FATLINK0TEX
#undef FATLINK1TEX
#undef LONGLINK0TEX
#undef LONGLINK1TEX
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_SPINOR
#undef ACCUMTEX
#undef READ_ACCUM
#undef CLOVERTEX
#undef READ_CLOVER
#undef DSLASH_CLOVER
#undef READ_FAT_MATRIX
#undef READ_LONG_MATRIX
#undef READ_1ST_NBR_SPINOR
#undef READ_3RD_NBR_SPINOR

// prepare next set of options, or clean up after final iteration

#if (DD_DAG==0)
#undef DD_DAG
#define DD_DAG 1
#else
#undef DD_DAG
#define DD_DAG 0

#if (DD_XPAY==0)
#undef DD_XPAY
#define DD_XPAY 1
#elif (DD_XPAY ==1)
#undef DD_XPAY
#define DD_XPAY 2
#else
#undef DD_XPAY
#define DD_XPAY 0

#if (DD_RECON==0)
#undef DD_RECON
#define DD_RECON 1
#elif (DD_RECON==1)
#undef DD_RECON
#define DD_RECON 2
#else
#undef DD_RECON
#define DD_RECON 0

#if (DD_GPREC==0)
#undef DD_GPREC
#define DD_GPREC 1
#elif (DD_GPREC==1)
#undef DD_GPREC
#define DD_GPREC 2
#else
#undef DD_GPREC
#define DD_GPREC 0

#if (DD_SPREC==0)
#undef DD_SPREC
#define DD_SPREC 1
#elif (DD_SPREC==1)
#undef DD_SPREC
#define DD_SPREC 2
#else
#undef DD_SPREC
#define DD_SPREC 0

//#if (DD_CPREC==0)
//#undef DD_CPREC
//#define DD_CPREC 1
#if (DD_CPREC==1)
#undef DD_CPREC
//#define DD_CPREC 2
//#elif (DD_CPREC==2)
//#undef DD_CPREC
#define DD_CPREC 3
#else

#undef DD_LOOP
#undef DD_DAG
#undef DD_XPAY
#undef DD_RECON
#undef DD_GPREC
#undef DD_SPREC
#undef DD_CPREC

#endif // DD_CPREC
#endif // DD_SPREC
#endif // DD_GPREC
#endif // DD_RECON
#endif // DD_XPAY
#endif // DD_DAG

#ifdef DD_LOOP
#include "dslash_def.h"
#endif
