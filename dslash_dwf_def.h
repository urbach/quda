//J  dslash_dwf_def.h 
//J  Ver. 09.07.c

//J  Dslash kernel definitions for DWF

//J  This file is structured so that
//J  the C preprocessor loops through 
//J  variants by recursively including
//J  itself.


//J  7/27/09:  continued hacking to turn of some precisions, for
//J     faster compile.
//J  7/29/09:  I think this file is ready to go now.

// As an example of the function naming conventions, consider
//
// dslashSHS12DaggerXpay_dwf_Kernel(float4* g_out, int oddBit, float a).
//
// This is a dwf Dslash^dagger kernel where the gauge field is read in single
// precision (S), the spinor field is read in half precision (H), the clover
// term is read in single precision (S), each gauge matrix is reconstructed
// from 12 real numbers, and the result is multiplied by "a" and summed
// with an input vector (Xpay).  More generally, each function name is given
// by the concatenation of the following 6 fields, with "dslash" at the
// beginning and "Kernel" at the end:
//
// DD_GPREC_F = D, S, H
// DD_SPREC_F = D, S, H

//J  Here I insert macros to turn off much of the
//J  recursive precision stuff, in order to get
//J  faster, more specialized builds.

//#define NO_D_PREC
//#define NO_S_PREC
#define NO_H_PREC
#define NO_CLOVER

// DD_CPREC_F = S, [blank]; the latter corresponds to plain Wilson
// DD_RECON_F = 12, 8
// DD_DAG_F = Dagger, [blank]
// DD_XPAY_F = Xpay, [blank]


// initialize on first iteration
#ifndef DD_LOOP
  #define DD_LOOP
  #define DD_DAG 0
  #define DD_XPAY 0
  #define DD_RECON 0
  #ifndef NO_D_PREC
    #define DD_GPREC 0
    #define DD_SPREC 0
  #else
    #ifndef NO_S_PREC
      #define DD_GPREC 1
      #define DD_SPREC 1
    #else
      #ifndef NO_H_PREC
        #define DD_GPREC 2
        #define DD_SPREC 2
      #else
        #error all precision levels turned off in dslash_dwf_def.h
      #endif
    #endif
  #endif
#endif

// set options for current iteration

#if (DD_DAG==0) // no dagger
  #define DD_DAG_F
#else           // dagger
  #define DD_DAG_F Dagger
#endif

//J  Set up the last 2 args that go into the ..._Kernel functions. 
#if (DD_XPAY==0) // no xpay 
  // Insert nothing into the kernel name.
  #define DD_XPAY_F
  // For DWF, we need the mass in the hopping terms.
  #if (DD_SPREC == 0)
    #define DD_PARAM2 int oddBit, double mferm
  #else
    #define DD_PARAM2 int oddBit, float mferm
  #endif
#else            // xpay
  // Insert "Xpay" into the kernel name.
  #define DD_XPAY_F Xpay
  //J  We will leave the 2nd arg a generic "a"
  //J  b/c when we get to even/odd preconditioning in the
  //J  inverter, we'll want -kappa^2 and whatnot.  On the
  //J  other hand, for naive dslash, it will be m0-5,
  //J  (or -(M5+5) in Andrew P.'s conventions)
  //J  the diagonal part of Shamir DWF.
  #if (DD_SPREC == 0)
    #define DD_PARAM2 int oddBit, double mferm, double a
  #else
    #define DD_PARAM2 int oddBit, float mferm, float a
  #endif
  //J  This is a flag that we're doing xpay.
  #define DSLASH_XPAY
#endif



//J  Harmless to leave this as is, since no
//J  cost for compilation.
#if (DD_RECON==0) // reconstruct from 12 reals
  #define DD_RECON_F 12
  #if (DD_GPREC==0)
    #define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_DOUBLE
    #define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_DOUBLE
  #elif (DD_GPREC==1)
    #define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE
    #define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_SINGLE
  #else
    #define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_12_SINGLE
    #define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_12_SINGLE
  #endif
#else             // reconstruct from 8 reals
  
  #define DD_RECON_F 8
  #if (DD_GPREC==0)
    #define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_DOUBLE
    #define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_DOUBLE
  #elif (DD_GPREC==1)
    #define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE
    #define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_SINGLE
  #else
    #define RECONSTRUCT_GAUGE_MATRIX RECONSTRUCT_MATRIX_8_SINGLE
    #define READ_GAUGE_MATRIX READ_GAUGE_MATRIX_8_HALF
  #endif
#endif

//J  Harmless to leave this as is, since no
//J  cost for compilation.
#if (DD_GPREC==0) // double-precision gauge field
  #define DD_GPREC_F D
  #define GAUGE0TEX gauge0TexDouble
  #define GAUGE1TEX gauge1TexDouble
#elif (DD_GPREC==1) // single-precision gauge field
  #define DD_GPREC_F S
  #define GAUGE0TEX gauge0TexSingle
  #define GAUGE1TEX gauge1TexSingle
#else             // half-precision gauge field
  #define DD_GPREC_F H
  #define GAUGE0TEX gauge0TexHalf
  #define GAUGE1TEX gauge1TexHalf
#endif

//J  Harmless to leave this as is, since no
//J  cost for compilation.
#if (DD_SPREC==0) // double-precision spinor field
  #define DD_SPREC_F D
  #define DD_PARAM1 double2* g_out
  #define READ_SPINOR READ_SPINOR_DOUBLE
  #define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
  #define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
  #define SPINORTEX spinorTexDouble
  #define WRITE_SPINOR WRITE_SPINOR_DOUBLE2
  #if (DD_XPAY==1)
    #define ACCUMTEX accumTexDouble
    #define READ_ACCUM READ_ACCUM_DOUBLE
  #endif
#elif (DD_SPREC==1) // single-precision spinor field
  #define DD_SPREC_F S
  #define DD_PARAM1 float4* g_out
  #define READ_SPINOR READ_SPINOR_SINGLE
  #define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
  #define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
  #define SPINORTEX spinorTexSingle
  #define WRITE_SPINOR WRITE_SPINOR_FLOAT4
  #if (DD_XPAY==1)
    #define ACCUMTEX accumTexSingle
    #define READ_ACCUM READ_ACCUM_SINGLE
  #endif
#else            // half-precision spinor field
  #define DD_SPREC_F H
  #define READ_SPINOR READ_SPINOR_HALF
  #define READ_SPINOR_UP READ_SPINOR_HALF_UP
  #define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
  #define SPINORTEX spinorTexHalf
  #define DD_PARAM1 short4* g_out, float *c
  #define WRITE_SPINOR WRITE_SPINOR_SHORT4
  #if (DD_XPAY==1)
    #define ACCUMTEX accumTexHalf
    #define READ_ACCUM READ_ACCUM_HALF
  #endif
#endif


#if !(__CUDA_ARCH__ != 130 && (DD_SPREC == 0 || DD_GPREC == 0))

  //J  Important modification here.
  #define DD_CONCAT(g,s,r,d,x) dslash ## g ## s ## r ## d ## x ## _dwf_Kernel
  #define DD_FUNC(g,s,r,d,x) DD_CONCAT(g,s,r,d,x)

  // define the kernel


  //J  Don't build precisions we've turned off.

  #define DO_NOTHING 0

  #ifdef NO_D_PREC
    #if (DD_GPREC == 0 || DD_SPREC == 0)
      #undef DO_NOTHING
      #define DO_NOTHING 1
    #endif
  #endif

  #ifdef NO_S_PREC
    #if (DD_GPREC == 1 || DD_SPREC == 1)
      #undef DO_NOTHING
      #define DO_NOTHING 1
    #endif
  #endif

  #ifdef NO_H_PREC
    #if (DD_GPREC == 2 || DD_SPREC == 2)
      #undef DO_NOTHING
      #define DO_NOTHING 1
    #endif
  #endif
  
  #if (DO_NOTHING == 0)
    __global__ void
    DD_FUNC(DD_GPREC_F, DD_SPREC_F, DD_RECON_F, DD_DAG_F, DD_XPAY_F)(DD_PARAM1, DD_PARAM2) {
    #if DD_DAG
      #include "dslash_dagger_core.h"
    #else
      #include "dslash_core.h"
    #endif
    }
  #endif

#endif  // CUDA_ARCH

// clean up

#undef DO_NOTHING
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
#undef READ_GAUGE_MATRIX
#undef RECONSTRUCT_GAUGE_MATRIX
#undef GAUGE0TEX
#undef GAUGE1TEX
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
  #else
    #undef DD_XPAY
    #define DD_XPAY 0
     
    #if (DD_RECON==0)
      #undef DD_RECON
      #define DD_RECON 1
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


          //J  I believe we only get here when the C preprocessing
          //J  loop is complete.
          #undef DD_LOOP  // 22 lines below, this terminates the C preprocessing loop.
          #undef DD_DAG
          #undef DD_XPAY
          #undef DD_RECON
          #undef DD_GPREC
          #undef DD_SPREC

          #undef DO_NOTHING
          //#undef NO_S_PREC
          //#undef NO_H_PREC




        #endif // DD_SPREC
      #endif // DD_GPREC
    #endif // DD_RECON
  #endif // DD_XPAY
#endif // DD_DAG

// Here's the recursive bit.
#ifdef DD_LOOP
  #include "dslash_dwf_def.h"
#endif


