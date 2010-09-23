// *** CUDA TWIST_DSLASH ***

//implementation of the following operators:
//1) R(a, b) * Dslash,
//where
//
//R(a, b) = a - i*b*gamma5, 
//a = 1 / (1 + (2*kappa*mu)^2), b = a * 2*kappa*mu; 
//
//2) T(a) + b * Dslash 
//
//where
//T(a) = 1 + i*a*gamma5,
//a = 2 * kappa * mu, b = - kappa*kappa

//Warning: parameter 'mu' already contains twisted mass flavor sign (predefined in the host functions), i.e. it is NOT the twisted mass

#define SHARED_FLOATS_PER_THREAD 8

// input spinor
#ifdef SPINOR_DOUBLE

#define spinorFloat double

#define i00_re I0.x
#define i00_im I0.y
#define i01_re I1.x
#define i01_im I1.y
#define i02_re I2.x
#define i02_im I2.y
#define i10_re I3.x
#define i10_im I3.y
#define i11_re I4.x
#define i11_im I4.y
#define i12_re I5.x
#define i12_im I5.y
#define i20_re I6.x
#define i20_im I6.y
#define i21_re I7.x
#define i21_im I7.y
#define i22_re I8.x
#define i22_im I8.y
#define i30_re I9.x
#define i30_im I9.y
#define i31_re I10.x
#define i31_im I10.y
#define i32_re I11.x
#define i32_im I11.y

#else

#define spinorFloat float

#define i00_re I0.x
#define i00_im I0.y
#define i01_re I0.z
#define i01_im I0.w
#define i02_re I1.x
#define i02_im I1.y
#define i10_re I1.z
#define i10_im I1.w
#define i11_re I2.x
#define i11_im I2.y
#define i12_re I2.z
#define i12_im I2.w
#define i20_re I3.x
#define i20_im I3.y
#define i21_re I3.z
#define i21_im I3.w
#define i22_re I4.x
#define i22_im I4.y
#define i30_re I4.z
#define i30_im I4.w
#define i31_re I5.x
#define i31_im I5.y
#define i32_re I5.z
#define i32_im I5.w

#endif // SPINOR_DOUBLE

// gauge link
#ifdef GAUGE_DOUBLE
#define g00_re G0.x
#define g00_im G0.y
#define g01_re G1.x
#define g01_im G1.y
#define g02_re G2.x
#define g02_im G2.y
#define g10_re G3.x
#define g10_im G3.y
#define g11_re G4.x
#define g11_im G4.y
#define g12_re G5.x
#define g12_im G5.y
#define g20_re G6.x
#define g20_im G6.y
#define g21_re G7.x
#define g21_im G7.y
#define g22_re G8.x
#define g22_im G8.y
// temporaries
#define A_re G9.x
#define A_im G9.y


#else
#define g00_re G0.x
#define g00_im G0.y
#define g01_re G0.z
#define g01_im G0.w
#define g02_re G1.x
#define g02_im G1.y
#define g10_re G1.z
#define g10_im G1.w
#define g11_re G2.x
#define g11_im G2.y
#define g12_re G2.z
#define g12_im G2.w
#define g20_re G3.x
#define g20_im G3.y
#define g21_re G3.z
#define g21_im G3.w
#define g22_re G4.x
#define g22_im G4.y
// temporaries
#define A_re G4.z
#define A_im G4.w

#endif // GAUGE_DOUBLE

// conjugated gauge link
#define gT00_re (+g00_re)
#define gT00_im (-g00_im)
#define gT01_re (+g10_re)
#define gT01_im (-g10_im)
#define gT02_re (+g20_re)
#define gT02_im (-g20_im)
#define gT10_re (+g01_re)
#define gT10_im (-g01_im)
#define gT11_re (+g11_re)
#define gT11_im (-g11_im)
#define gT12_re (+g21_re)
#define gT12_im (-g21_im)
#define gT20_re (+g02_re)
#define gT20_im (-g02_im)
#define gT21_re (+g12_re)
#define gT21_im (-g12_im)
#define gT22_re (+g22_re)
#define gT22_im (-g22_im)

// output spinor
#define o00_re s[0*SHARED_STRIDE]
#define o00_im s[1*SHARED_STRIDE]
#define o01_re s[2*SHARED_STRIDE]
#define o01_im s[3*SHARED_STRIDE]
#define o02_re s[4*SHARED_STRIDE]
#define o02_im s[5*SHARED_STRIDE]
#define o10_re s[6*SHARED_STRIDE]
#define o10_im s[7*SHARED_STRIDE]
volatile spinorFloat o11_re;
volatile spinorFloat o11_im;
volatile spinorFloat o12_re;
volatile spinorFloat o12_im;
volatile spinorFloat o20_re;
volatile spinorFloat o20_im;
volatile spinorFloat o21_re;
volatile spinorFloat o21_im;
volatile spinorFloat o22_re;
volatile spinorFloat o22_im;
volatile spinorFloat o30_re;
volatile spinorFloat o30_im;
volatile spinorFloat o31_re;
volatile spinorFloat o31_im;
volatile spinorFloat o32_re;
volatile spinorFloat o32_im;



#include "read_gauge.h"
#include "read_clover.h"
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
int z1 = FAST_INT_DIVIDE(sid, X1h);
int x1h = sid - z1*X1h;
int z2 = FAST_INT_DIVIDE(z1, X2);
int x2 = z1 - z2*X2;
int x4 = FAST_INT_DIVIDE(z2, X3);
int x3 = z2 - x4*X3;
int x1odd = (x2 + x3 + x4 + oddBit) & 1;
int x1 = 2*x1h + x1odd;
int X = 2*sid + x1odd;

#ifdef SPINOR_DOUBLE
#define SHARED_STRIDE 8  // to avoid bank conflicts
extern __shared__ spinorFloat sd_data[];
volatile spinorFloat *s = sd_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts
extern __shared__ spinorFloat ss_data[];
volatile spinorFloat *s = ss_data + SHARED_FLOATS_PER_THREAD*SHARED_STRIDE*(threadIdx.x/SHARED_STRIDE)
                                  + (threadIdx.x % SHARED_STRIDE);
#endif

spinorFloat a, b;

#ifndef DSLASH_XPAY//set paramters for tm_dslash{...}Kernel

  b = 2 * kappa * mu;//warning: 'mu' contains twisted mass flavor sign (it is not the twisted mass)
  a = 1 / (1 + b * b);
  b *= a;

#else//set parameters for tm_dslashXpay{...}Kernel

  a = 2 * kappa * mu;//warning: 'mu' contains twisted mass flavor sign (it is not the twisted mass)
  b = - kappa * kappa;

#endif


o00_re = o00_im = 0;
o01_re = o01_im = 0;
o02_re = o02_im = 0;
o10_re = o10_im = 0;
o11_re = o11_im = 0;
o12_re = o12_im = 0;
o20_re = o20_im = 0;
o21_re = o21_im = 0;
o22_re = o22_im = 0;
o30_re = o30_im = 0;
o31_re = o31_im = 0;
o32_re = o32_im = 0;

{
    // Projector R(a, b) * P0- 
    // a	b 	-i*b 	-i*a 
    // b 	a 	-i*a 	-i*b 
    // -i*b 	i*a 	a 	-b 
    // i*a 	-i*b 	-b 	a 
    
    int sp_idx = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 0);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(0);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i30_im;
    spinorFloat a0_im = +i00_im-i30_re;
    spinorFloat a1_re = +i01_re+i31_im;
    spinorFloat a1_im = +i01_im-i31_re;
    spinorFloat a2_re = +i02_re+i32_im;
    spinorFloat a2_im = +i02_im-i32_re;
    
    spinorFloat b0_re = +i10_re+i20_im;
    spinorFloat b0_im = +i10_im-i20_re;
    spinorFloat b1_re = +i11_re+i21_im;
    spinorFloat b1_im = +i11_im-i21_re;
    spinorFloat b2_re = +i12_re+i22_im;
    spinorFloat b2_im = +i12_im-i22_re;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
 #ifndef DSLASH_XPAY
    
    o00_re += a * A0_re + b * B0_re;
    o00_im += a * A0_im + b * B0_im;
    o10_re -= -b * A0_re - a * B0_re;
    o10_im -= -b * A0_im - a * B0_im;
    o20_re -= -b * A0_im + a * B0_im;
    o20_im += -b * A0_re + a * B0_re;
    o30_re -= a * A0_im - b * B0_im;
    o30_im += a * A0_re - b * B0_re;
    
    o01_re += a * A1_re + b * B1_re;
    o01_im += a * A1_im + b * B1_im;
    o11_re -= -b * A1_re - a * B1_re;
    o11_im -= -b * A1_im - a * B1_im;
    o21_re -= -b * A1_im + a * B1_im;
    o21_im += -b * A1_re + a * B1_re;
    o31_re -= a * A1_im - b * B1_im;
    o31_im += a * A1_re - b * B1_re;
    
    o02_re += a * A2_re + b * B2_re;
    o02_im += a * A2_im + b * B2_im;
    o12_re -= -b * A2_re - a * B2_re;
    o12_im -= -b * A2_im - a * B2_im;
    o22_re -= -b * A2_im + a * B2_im;
    o22_im += -b * A2_re + a * B2_re;
    o32_re -= a * A2_im - b * B2_im;
    o32_im += a * A2_re - b * B2_re;
 
//Note: additional 24 * 3 = 72 fp operations w.r.t. standard dslash    
    
#else //DSLASH_XPAY

    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= B0_im;
    o20_im += B0_re;
    o30_re -= A0_im;
    o30_im += A0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= B1_im;
    o21_im += B1_re;
    o31_re -= A1_im;
    o31_im += A1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= B2_im;
    o22_im += B2_re;
    o32_re -= A2_im;
    o32_im += A2_re;
 
#endif    
}

{
    // Projector R(a,b) * P0+
    // a	-b 	-i*b	 i*a 
    // -b 	a 	i*a 	-i*b 
    // -i*b 	-i*a	a 	+b 
    // -i*a -	i*b 	+b 	a 
    
    int sp_idx = ((x1==0)    ? X+X1m1 : X-1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 1);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(1);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i30_im;
    spinorFloat a0_im = +i00_im+i30_re;
    spinorFloat a1_re = +i01_re-i31_im;
    spinorFloat a1_im = +i01_im+i31_re;
    spinorFloat a2_re = +i02_re-i32_im;
    spinorFloat a2_im = +i02_im+i32_re;
    
    spinorFloat b0_re = +i10_re-i20_im;
    spinorFloat b0_im = +i10_im+i20_re;
    spinorFloat b1_re = +i11_re-i21_im;
    spinorFloat b1_im = +i11_im+i21_re;
    spinorFloat b2_re = +i12_re-i22_im;
    spinorFloat b2_im = +i12_im+i22_re;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
  
#ifndef DSLASH_XPAY

    o00_re += a * A0_re - b * B0_re;
    o00_im += a * A0_im - b * B0_im;
    o10_re += -b * A0_re + a * B0_re;
    o10_im += -b * A0_im + a * B0_im;
    o20_re -= -b * A0_im - a * B0_im;
    o20_im += -b * A0_re - a * B0_re;
    o30_re -= -a * A0_im - b * B0_im;
    o30_im += -a * A0_re - b * B0_re;
    
    o01_re += a * A1_re - b * B1_re;
    o01_im += a * A1_im - b * B1_im;
    o11_re += -b * A1_re + a * B1_re;
    o11_im += -b * A1_im + a * B1_im;
    o21_re -= -b * A1_im - a * B1_im;
    o21_im += -b * A1_re - a * B1_re;
    o31_re -= -a * A1_im - b * B1_im;
    o31_im += -a * A1_re - b * B1_re;
    
    o02_re += a * A2_re - b * B2_re;
    o02_im += a * A2_im - b * B2_im;
    o12_re += -b * A2_re + a * B2_re;
    o12_im += -b * A2_im + a * B2_im;
    o22_re -= -b * A2_im - a * B2_im;
    o22_im += -b * A2_re - a * B2_re;
    o32_re -= -a * A2_im - b * B2_im;
    o32_im += -a * A2_re - b * B2_re;
    
//Note: additional 24 * 3 = 72 fp operations w.r.t. standard dslash     
    
#else    
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += B0_im;
    o20_im -= B0_re;
    o30_re += A0_im;
    o30_im -= A0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += B1_im;
    o21_im -= B1_re;
    o31_re += A1_im;
    o31_im -= A1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += B2_im;
    o22_im -= B2_re;
    o32_re += A2_im;
    o32_im -= A2_re;
    
#endif    
   
}

{
    // Projector R(a,b) * P1-
    // a 	-i*b 	-i*b 	-a 
    // i*b 	a 	a 	-i*b 
    // -i*b 	a 	a 	i*b 
    // -a 	-i*b 	-i*b 	a 
    
    int sp_idx = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 2);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(2);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i30_re;
    spinorFloat a0_im = +i00_im-i30_im;
    spinorFloat a1_re = +i01_re-i31_re;
    spinorFloat a1_im = +i01_im-i31_im;
    spinorFloat a2_re = +i02_re-i32_re;
    spinorFloat a2_im = +i02_im-i32_im;
    
    spinorFloat b0_re = +i10_re+i20_re;
    spinorFloat b0_im = +i10_im+i20_im;
    spinorFloat b1_re = +i11_re+i21_re;
    spinorFloat b1_im = +i11_im+i21_im;
    spinorFloat b2_re = +i12_re+i22_re;
    spinorFloat b2_im = +i12_im+i22_im;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
#ifndef DSLASH_XPAY    
    
    o00_re += a * A0_re + b * B0_im;
    o00_im += a * A0_im - b * B0_re;
    o10_re += -b * A0_im + a * B0_re;
    o10_im -= -b * A0_re - a * B0_im;
    o20_re -= -b * A0_im - a * B0_re;
    o20_im += -b * A0_re + a * B0_im;
    o30_re -= a * A0_re - b * B0_im;
    o30_im -= a * A0_im + b * B0_re;
    
    o01_re += a * A1_re + b * B1_im;
    o01_im += a * A1_im - b * B1_re;
    o11_re += -b * A1_im + a * B1_re;
    o11_im -= -b * A1_re - a * B1_im;
    o21_re -= -b * A1_im - a * B1_re;
    o21_im += -b * A1_re + a * B1_im;
    o31_re -= a * A1_re - b * B1_im;
    o31_im -= a * A1_im + b * B1_re;
    
    o02_re += a * A2_re + b * B2_im;
    o02_im += a * A2_im - b * B2_re;
    o12_re += -b * A2_im + a * B2_re;
    o12_im -= -b * A2_re - a * B2_im;
    o22_re -= -b * A2_im - a * B2_re;
    o22_im += -b * A2_re + a * B2_im;
    o32_re -= a * A2_re - b * B2_im;
    o32_im -= a * A2_im + b * B2_re;
    
//Note: additional 24 * 3 = 72 fp operations w.r.t. standard dslash     
    
#else

    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += B0_re;
    o20_im += B0_im;
    o30_re -= A0_re;
    o30_im -= A0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += B1_re;
    o21_im += B1_im;
    o31_re -= A1_re;
    o31_im -= A1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += B2_re;
    o22_im += B2_im;
    o32_re -= A2_re;
    o32_im -= A2_im;

#endif
    
}

{
    // Projector R(a,b) * P1+
    // a 	i*b 	-i*b 	a 
    // -i*b 	a 	-a 	-i*b 
    // -i*b 	-a 	a 	-i*b 
    // a 	-i*b 	i*b 	a 
    
    int sp_idx = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 3);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(3);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i30_re;
    spinorFloat a0_im = +i00_im+i30_im;
    spinorFloat a1_re = +i01_re+i31_re;
    spinorFloat a1_im = +i01_im+i31_im;
    spinorFloat a2_re = +i02_re+i32_re;
    spinorFloat a2_im = +i02_im+i32_im;
    
    spinorFloat b0_re = +i10_re-i20_re;
    spinorFloat b0_im = +i10_im-i20_im;
    spinorFloat b1_re = +i11_re-i21_re;
    spinorFloat b1_im = +i11_im-i21_im;
    spinorFloat b2_re = +i12_re-i22_re;
    spinorFloat b2_im = +i12_im-i22_im;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
#ifndef DSLASH_XPAY
    
    o00_re += a * A0_re - b * B0_im;
    o00_im += a * A0_im + b * B0_re;
    o10_re -= -b * A0_im - a * B0_re;
    o10_im += -b * A0_re + a * B0_im;
    o20_re -= -b * A0_im + a * B0_re;
    o20_im += -b * A0_re - a * B0_im;
    o30_re += a * A0_re + b * B0_im;
    o30_im += a * A0_im - b * B0_re;
    
    o01_re += a * A1_re - b * B1_im;
    o01_im += a * A1_im + b * B1_re;
    o11_re -= -b * A1_im - a * B1_re;
    o11_im += -b * A1_re + a * B1_im;
    o21_re -= -b * A1_im + a * B1_re;
    o21_im += -b * A1_re - a * B1_im;
    o31_re += a * A1_re + b * B1_im;
    o31_im += a * A1_im - b * B1_re;
    
    o02_re += a * A2_re - b * B2_im;
    o02_im += a * A2_im + b * B2_re;
    o12_re -= -b * A2_im - a * B2_re;
    o12_im += -b * A2_re + a * B2_im;
    o22_re -= -b * A2_im + a * B2_re;
    o22_im += -b * A2_re - a * B2_im;
    o32_re += a * A2_re + b * B2_im;
    o32_im += a * A2_im - b * B2_re;

//Note: additional 24 * 3 = 72 fp operations w.r.t. standard dslash     
    
#else    
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= B0_re;
    o20_im -= B0_im;
    o30_re += A0_re;
    o30_im += A0_im;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= B1_re;
    o21_im -= B1_im;
    o31_re += A1_re;
    o31_im += A1_im;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= B2_re;
    o22_im -= B2_im;
    o32_re += A2_re;
    o32_im += A2_im;
    
#endif    
}

{
    // Projector  R(a,b) * P2-
    // (a+b) 	0 	-i(a+b) 	0 
    // 0 	(a-b) 	0 		i(a-b) 
    // i(a-b) 0 	(a-b) 		0 
    // 0 	-i(a+b) 0 		(a+b) 
    
    int sp_idx = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE0TEX, 4);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(4);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re+i20_im;
    spinorFloat a0_im = +i00_im-i20_re;
    spinorFloat a1_re = +i01_re+i21_im;
    spinorFloat a1_im = +i01_im-i21_re;
    spinorFloat a2_re = +i02_re+i22_im;
    spinorFloat a2_im = +i02_im-i22_re;
    
    spinorFloat b0_re = +i10_re-i30_im;
    spinorFloat b0_im = +i10_im+i30_re;
    spinorFloat b1_re = +i11_re-i31_im;
    spinorFloat b1_im = +i11_im+i31_re;
    spinorFloat b2_re = +i12_re-i32_im;
    spinorFloat b2_im = +i12_im+i32_re;
    
    // multiply row 0
    spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
    spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
    spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
    spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
    spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
    spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
    spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
    spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
    spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
    spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
    
#ifndef DSLASH_XPAY

    spinorFloat apb = a + b;
    spinorFloat amb = a - b;

    o00_re += apb * A0_re;
    o00_im += apb * A0_im;
    o10_re += amb * B0_re;
    o10_im += amb * B0_im;
    o20_re -= amb * A0_im;
    o20_im += amb * A0_re;
    o30_re += apb * B0_im;
    o30_im -= apb * B0_re;
    
    o01_re += apb * A1_re;
    o01_im += apb * A1_im;
    o11_re += amb * B1_re;
    o11_im += amb * B1_im;
    o21_re -= amb * A1_im;
    o21_im += amb * A1_re;
    o31_re += apb * B1_im;
    o31_im -= apb * B1_re;
    
    o02_re += apb * A2_re;
    o02_im += apb * A2_im;
    o12_re += amb * B2_re;
    o12_im += amb * B2_im;
    o22_re -= amb * A2_im;
    o22_im += amb * A2_re;
    o32_re += apb * B2_im;
    o32_im -= apb * B2_re;
    
//Note: additional 24 fp operations w.r.t. standard dslash     
    
#else    
    
    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re -= A0_im;
    o20_im += A0_re;
    o30_re += B0_im;
    o30_im -= B0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re -= A1_im;
    o21_im += A1_re;
    o31_re += B1_im;
    o31_im -= B1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re -= A2_im;
    o22_im += A2_re;
    o32_re += B2_im;
    o32_im -= B2_re;
    
#endif    
}

{
    // Projector R(a,b) * P2+
    // a-b 	0 	i(a-b) 	0 
    // 0 	(a+b) 	0 	-i(a+b) 
    // -i(a+b) 0 	a+b 	0 
    // 0 	i(a-b) 0 	a-b 
    
    int sp_idx = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(GAUGE1TEX, 5);
    
    // read spinor from device memory
    READ_SPINOR(SPINORTEX);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(5);
    
    // project spinor into half spinors
    spinorFloat a0_re = +i00_re-i20_im;
    spinorFloat a0_im = +i00_im+i20_re;
    spinorFloat a1_re = +i01_re-i21_im;
    spinorFloat a1_im = +i01_im+i21_re;
    spinorFloat a2_re = +i02_re-i22_im;
    spinorFloat a2_im = +i02_im+i22_re;
    
    spinorFloat b0_re = +i10_re+i30_im;
    spinorFloat b0_im = +i10_im-i30_re;
    spinorFloat b1_re = +i11_re+i31_im;
    spinorFloat b1_im = +i11_im-i31_re;
    spinorFloat b2_re = +i12_re+i32_im;
    spinorFloat b2_im = +i12_im-i32_re;
    
    // multiply row 0
    spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
    spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
    spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
    spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
    
    // multiply row 1
    spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
    spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
    spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
    spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
    
    // multiply row 2
    spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
    spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
    spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
    spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);
    
#ifndef DSLASH_XPAY

    spinorFloat apb = a + b;
    spinorFloat amb = a - b;
    
    o00_re += amb * A0_re;
    o00_im += amb * A0_im;
    o10_re += apb * B0_re;
    o10_im += apb * B0_im;
    o20_re += apb * A0_im;
    o20_im -= apb * A0_re;
    o30_re -= amb * B0_im;
    o30_im += amb * B0_re;
    
    o01_re += amb * A1_re;
    o01_im += amb * A1_im;
    o11_re += apb * B1_re;
    o11_im += apb * B1_im;
    o21_re += apb * A1_im;
    o21_im -= apb * A1_re;
    o31_re -= amb * B1_im;
    o31_im += amb * B1_re;
    
    o02_re += amb * A2_re;
    o02_im += amb * A2_im;
    o12_re += apb * B2_re;
    o12_im += apb * B2_im;
    o22_re += apb * A2_im;
    o22_im -= apb * A2_re;
    o32_re -= amb * B2_im;
    o32_im += amb * B2_re;
    
//Note: additional 24 = 48 fp operations w.r.t. standard dslash     
    
#else 

    o00_re += A0_re;
    o00_im += A0_im;
    o10_re += B0_re;
    o10_im += B0_im;
    o20_re += A0_im;
    o20_im -= A0_re;
    o30_re -= B0_im;
    o30_im += B0_re;
    
    o01_re += A1_re;
    o01_im += A1_im;
    o11_re += B1_re;
    o11_im += B1_im;
    o21_re += A1_im;
    o21_im -= A1_re;
    o31_re -= B1_im;
    o31_im += B1_re;
    
    o02_re += A2_re;
    o02_im += A2_im;
    o12_re += B2_re;
    o12_im += B2_im;
    o22_re += A2_im;
    o22_im -= A2_re;
    o32_re -= B2_im;
    o32_im += B2_re;
    
#endif    
}
{
    // Projector R(a, b) * P3-
    // 0 0 -i*2 *b 0 
    // 0 0 0 -i*2*b 
    // 0 0 2*a 0 
    // 0 0 0 2*a 
    
    int sp_idx = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
    int ga_idx = sid;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i20_re;
        spinorFloat a0_im = +2*i20_im;
        spinorFloat a1_re = +2*i21_re;
        spinorFloat a1_im = +2*i21_im;
        spinorFloat a2_re = +2*i22_re;
        spinorFloat a2_im = +2*i22_im;
        
        spinorFloat b0_re = +2*i30_re;
        spinorFloat b0_im = +2*i30_im;
        spinorFloat b1_re = +2*i31_re;
        spinorFloat b1_im = +2*i31_im;
        spinorFloat b2_re = +2*i32_re;
        spinorFloat b2_im = +2*i32_im;
        
        // identity gauge matrix
        spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
        spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
        spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
        spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
        spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
        spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;

#ifndef DSLASH_XPAY	
	
	o00_re += b * A0_im;
        o00_im -= b * A0_re;
        o10_re += b * B0_im;
        o10_im -= b * B0_re;
        o20_re += a * A0_re;
        o20_im += a * A0_im;
        o30_re += a * B0_re;
        o30_im += a * B0_im;
	
	o01_re += b * A1_im;
        o01_im -= b * A1_re;
        o11_re += b * B1_im;
        o11_im -= b * B1_re;
        o21_re += a * A1_re;
        o21_im += a * A1_im;
        o31_re += a * B1_re;
        o31_im += a * B1_im;
        
	o02_re += b * A2_im;
        o02_im -= b * A2_re;
        o12_re += b * B2_im;
        o12_im -= b * B2_re;
        o22_re += a * A2_re;
        o22_im += a * A2_im;
        o32_re += a * B2_re;
        o32_im += a * B2_im;
	
//Note: additional 12 + 12 * 2 = 36 fp operations w.r.t. standard dslash 	
	
#else	
	
	o20_re += A0_re;
        o20_im += A0_im;
        o30_re += B0_re;
        o30_im += B0_im;
        
        o21_re += A1_re;
        o21_im += A1_im;
        o31_re += B1_re;
        o31_im += B1_im;
        
        o22_re += A2_re;
        o22_im += A2_im;
        o32_re += B2_re;
        o32_im += B2_im;
#endif        
    }
    else {
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE0TEX, 6);
        
        // read spinor from device memory
        READ_SPINOR_DOWN(SPINORTEX);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(6);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i20_re;
        spinorFloat a0_im = +2*i20_im;
        spinorFloat a1_re = +2*i21_re;
        spinorFloat a1_im = +2*i21_im;
        spinorFloat a2_re = +2*i22_re;
        spinorFloat a2_im = +2*i22_im;
        
        spinorFloat b0_re = +2*i30_re;
        spinorFloat b0_im = +2*i30_im;
        spinorFloat b1_re = +2*i31_re;
        spinorFloat b1_im = +2*i31_im;
        spinorFloat b2_re = +2*i32_re;
        spinorFloat b2_im = +2*i32_im;
        
        // multiply row 0
        spinorFloat A0_re = + (g00_re * a0_re - g00_im * a0_im) + (g01_re * a1_re - g01_im * a1_im) + (g02_re * a2_re - g02_im * a2_im);
        spinorFloat A0_im = + (g00_re * a0_im + g00_im * a0_re) + (g01_re * a1_im + g01_im * a1_re) + (g02_re * a2_im + g02_im * a2_re);
        spinorFloat B0_re = + (g00_re * b0_re - g00_im * b0_im) + (g01_re * b1_re - g01_im * b1_im) + (g02_re * b2_re - g02_im * b2_im);
        spinorFloat B0_im = + (g00_re * b0_im + g00_im * b0_re) + (g01_re * b1_im + g01_im * b1_re) + (g02_re * b2_im + g02_im * b2_re);
        
        // multiply row 1
        spinorFloat A1_re = + (g10_re * a0_re - g10_im * a0_im) + (g11_re * a1_re - g11_im * a1_im) + (g12_re * a2_re - g12_im * a2_im);
        spinorFloat A1_im = + (g10_re * a0_im + g10_im * a0_re) + (g11_re * a1_im + g11_im * a1_re) + (g12_re * a2_im + g12_im * a2_re);
        spinorFloat B1_re = + (g10_re * b0_re - g10_im * b0_im) + (g11_re * b1_re - g11_im * b1_im) + (g12_re * b2_re - g12_im * b2_im);
        spinorFloat B1_im = + (g10_re * b0_im + g10_im * b0_re) + (g11_re * b1_im + g11_im * b1_re) + (g12_re * b2_im + g12_im * b2_re);
        
        // multiply row 2
        spinorFloat A2_re = + (g20_re * a0_re - g20_im * a0_im) + (g21_re * a1_re - g21_im * a1_im) + (g22_re * a2_re - g22_im * a2_im);
        spinorFloat A2_im = + (g20_re * a0_im + g20_im * a0_re) + (g21_re * a1_im + g21_im * a1_re) + (g22_re * a2_im + g22_im * a2_re);
        spinorFloat B2_re = + (g20_re * b0_re - g20_im * b0_im) + (g21_re * b1_re - g21_im * b1_im) + (g22_re * b2_re - g22_im * b2_im);
        spinorFloat B2_im = + (g20_re * b0_im + g20_im * b0_re) + (g21_re * b1_im + g21_im * b1_re) + (g22_re * b2_im + g22_im * b2_re);
        
#ifndef DSLASH_XPAY	
	
	o00_re += b * A0_im;
        o00_im -= b * A0_re;
        o10_re += b * B0_im;
        o10_im -= b * B0_re;
        o20_re += a * A0_re;
        o20_im += a * A0_im;
        o30_re += a * B0_re;
        o30_im += a * B0_im;
	
	o01_re += b * A1_im;
        o01_im -= b * A1_re;
        o11_re += b * B1_im;
        o11_im -= b * B1_re;
        o21_re += a * A1_re;
        o21_im += a * A1_im;
        o31_re += a * B1_re;
        o31_im += a * B1_im;
        
	o02_re += b * A2_im;
        o02_im -= b * A2_re;
        o12_re += b * B2_im;
        o12_im -= b * B2_re;
        o22_re += a * A2_re;
        o22_im += a * A2_im;
        o32_re += a * B2_re;
        o32_im += a * B2_im;
	
//Note: additional 12 + 12 * 2 = 36 fp operations w.r.t. standard dslash 	
	
#else

        o20_re += A0_re;
        o20_im += A0_im;
        o30_re += B0_re;
        o30_im += B0_im;
        
        o21_re += A1_re;
        o21_im += A1_im;
        o31_re += B1_re;
        o31_im += B1_im;
        
        o22_re += A2_re;
        o22_im += A2_im;
        o32_re += B2_re;
        o32_im += B2_im;

#endif
        
    }
}

{
    // Projector R(a, b) * P3+
    // 2*a 0 0 0 
    // 0 2*a 0 0 
    // -i*2*b 0 0 0 
    // 0 -i*2*b 0 0 
    
    int sp_idx = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
    int ga_idx = sp_idx;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i00_re;
        spinorFloat a0_im = +2*i00_im;
        spinorFloat a1_re = +2*i01_re;
        spinorFloat a1_im = +2*i01_im;
        spinorFloat a2_re = +2*i02_re;
        spinorFloat a2_im = +2*i02_im;
        
        spinorFloat b0_re = +2*i10_re;
        spinorFloat b0_im = +2*i10_im;
        spinorFloat b1_re = +2*i11_re;
        spinorFloat b1_im = +2*i11_im;
        spinorFloat b2_re = +2*i12_re;
        spinorFloat b2_im = +2*i12_im;
        
        // identity gauge matrix
        spinorFloat A0_re = a0_re; spinorFloat A0_im = a0_im;
        spinorFloat B0_re = b0_re; spinorFloat B0_im = b0_im;
        spinorFloat A1_re = a1_re; spinorFloat A1_im = a1_im;
        spinorFloat B1_re = b1_re; spinorFloat B1_im = b1_im;
        spinorFloat A2_re = a2_re; spinorFloat A2_im = a2_im;
        spinorFloat B2_re = b2_re; spinorFloat B2_im = b2_im;
        
#ifndef DSLASH_XPAY        
	
	o00_re += a * A0_re;
        o00_im += a * A0_im;
        o10_re += a * B0_re;
        o10_im += a * B0_im;
        o20_re += b * A0_im;
        o20_im -= b * A0_re;
        o30_re += b * B0_im;
        o30_im -= b * B0_re;	
        
	o01_re += a * A1_re;
        o01_im += a * A1_im;
        o11_re += a * B1_re;
        o11_im += a * B1_im;
        o21_re += b * A1_im;
        o21_im -= b * A1_re;
        o31_re += b * B1_im;
        o31_im -= b * B1_re;
        
	o02_re += a * A2_re;
        o02_im += a * A2_im;
        o12_re += a * B2_re;
        o12_im += a * B2_im;
        o22_re += b * A2_im;
        o22_im -= b * A2_re;
        o32_re += b * B2_im;
        o32_im -= b * B2_re;
	
//Note: additional 12 + 12 * 2 = 36 fp operations w.r.t. standard dslash 	
	
#else

        o00_re += A0_re;
        o00_im += A0_im;
        o10_re += B0_re;
        o10_im += B0_im;
        
        o01_re += A1_re;
        o01_im += A1_im;
        o11_re += B1_re;
        o11_im += B1_im;
        
        o02_re += A2_re;
        o02_im += A2_im;
        o12_re += B2_re;
        o12_im += B2_im;
 
#endif 
    }
    else {
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(GAUGE1TEX, 7);
        
        // read spinor from device memory
        READ_SPINOR_UP(SPINORTEX);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(7);
        
        // project spinor into half spinors
        spinorFloat a0_re = +2*i00_re;
        spinorFloat a0_im = +2*i00_im;
        spinorFloat a1_re = +2*i01_re;
        spinorFloat a1_im = +2*i01_im;
        spinorFloat a2_re = +2*i02_re;
        spinorFloat a2_im = +2*i02_im;
        
        spinorFloat b0_re = +2*i10_re;
        spinorFloat b0_im = +2*i10_im;
        spinorFloat b1_re = +2*i11_re;
        spinorFloat b1_im = +2*i11_im;
        spinorFloat b2_re = +2*i12_re;
        spinorFloat b2_im = +2*i12_im;
        
        // multiply row 0
        spinorFloat A0_re = + (gT00_re * a0_re - gT00_im * a0_im) + (gT01_re * a1_re - gT01_im * a1_im) + (gT02_re * a2_re - gT02_im * a2_im);
        spinorFloat A0_im = + (gT00_re * a0_im + gT00_im * a0_re) + (gT01_re * a1_im + gT01_im * a1_re) + (gT02_re * a2_im + gT02_im * a2_re);
        spinorFloat B0_re = + (gT00_re * b0_re - gT00_im * b0_im) + (gT01_re * b1_re - gT01_im * b1_im) + (gT02_re * b2_re - gT02_im * b2_im);
        spinorFloat B0_im = + (gT00_re * b0_im + gT00_im * b0_re) + (gT01_re * b1_im + gT01_im * b1_re) + (gT02_re * b2_im + gT02_im * b2_re);
        
        // multiply row 1
        spinorFloat A1_re = + (gT10_re * a0_re - gT10_im * a0_im) + (gT11_re * a1_re - gT11_im * a1_im) + (gT12_re * a2_re - gT12_im * a2_im);
        spinorFloat A1_im = + (gT10_re * a0_im + gT10_im * a0_re) + (gT11_re * a1_im + gT11_im * a1_re) + (gT12_re * a2_im + gT12_im * a2_re);
        spinorFloat B1_re = + (gT10_re * b0_re - gT10_im * b0_im) + (gT11_re * b1_re - gT11_im * b1_im) + (gT12_re * b2_re - gT12_im * b2_im);
        spinorFloat B1_im = + (gT10_re * b0_im + gT10_im * b0_re) + (gT11_re * b1_im + gT11_im * b1_re) + (gT12_re * b2_im + gT12_im * b2_re);
        
        // multiply row 2
        spinorFloat A2_re = + (gT20_re * a0_re - gT20_im * a0_im) + (gT21_re * a1_re - gT21_im * a1_im) + (gT22_re * a2_re - gT22_im * a2_im);
        spinorFloat A2_im = + (gT20_re * a0_im + gT20_im * a0_re) + (gT21_re * a1_im + gT21_im * a1_re) + (gT22_re * a2_im + gT22_im * a2_re);
        spinorFloat B2_re = + (gT20_re * b0_re - gT20_im * b0_im) + (gT21_re * b1_re - gT21_im * b1_im) + (gT22_re * b2_re - gT22_im * b2_im);
        spinorFloat B2_im = + (gT20_re * b0_im + gT20_im * b0_re) + (gT21_re * b1_im + gT21_im * b1_re) + (gT22_re * b2_im + gT22_im * b2_re);

#ifndef DSLASH_XPAY	
	
	o00_re += a * A0_re;
        o00_im += a * A0_im;
        o10_re += a * B0_re;
        o10_im += a * B0_im;
        o20_re += b * A0_im;
        o20_im -= b * A0_re;
        o30_re += b * B0_im;
        o30_im -= b * B0_re;	
        
	o01_re += a * A1_re;
        o01_im += a * A1_im;
        o11_re += a * B1_re;
        o11_im += a * B1_im;
        o21_re += b * A1_im;
        o21_im -= b * A1_re;
        o31_re += b * B1_im;
        o31_im -= b * B1_re;
        
	o02_re += a * A2_re;
        o02_im += a * A2_im;
        o12_re += a * B2_re;
        o12_im += a * B2_im;
        o22_re += b * A2_im;
        o22_im -= b * A2_re;
        o32_re += b * B2_im;
        o32_im -= b * B2_re;
	
//Note: additional 12 + 12 * 2 = 36 fp operations w.r.t. standard dslash 	
	
#else

        o00_re += A0_re;
        o00_im += A0_im;
        o10_re += B0_re;
        o10_im += B0_im;
        
        o01_re += A1_re;
        o01_im += A1_im;
        o11_re += B1_re;
        o11_im += B1_im;
        
        o02_re += A2_re;
        o02_im += A2_im;
        o12_re += B2_re;
        o12_im += B2_im;
 
#endif 	
        
    }
}

//Note: in total 408 + 10 = 412 additional fp operations for twisted mass dslash

#ifdef DSLASH_XPAY
    READ_ACCUM(ACCUMTEX)
#ifdef SPINOR_DOUBLE

#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp1.x
#define tmp1_im tmp1.y
#define tmp2_re tmp2.x
#define tmp2_im tmp2.y
#define tmp3_re tmp3.x
#define tmp3_im tmp3.y

    double2 tmp0, tmp1, tmp2, tmp3;
    
    //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
    
    //get the 1st color component:
    
    tmp0_re = accum0.x - a * accum6.y;
    tmp0_im = accum0.y + a * accum6.x;
    
    tmp2_re = accum6.x - a * accum0.y;
    tmp2_im = accum6.y + a * accum0.x;

    tmp1_re = accum3.x - a * accum9.y;
    tmp1_im = accum3.y + a * accum9.x;
    
    tmp3_re = accum9.x - a * accum3.y;
    tmp3_im = accum9.y + a * accum3.x;
    
    o00_re = b * o00_re + tmp0_re;
    o00_im = b * o00_im + tmp0_im;
    o10_re = b * o10_re + tmp1_re;
    o10_im = b * o10_im + tmp1_im;
    o20_re = b * o20_re + tmp2_re;
    o20_im = b * o20_im + tmp2_im;
    o30_re = b * o30_re + tmp3_re;
    o30_im = b * o30_im + tmp3_im;
    
    //get the 2nd color component:    
    
    tmp0_re = accum1.x - a * accum7.y;
    tmp0_im = accum1.y + a * accum7.x;
    
    tmp2_re = accum7.x - a * accum1.y;
    tmp2_im = accum7.y + a * accum1.x;

    tmp1_re = accum4.x - a * accum10.y;
    tmp1_im = accum4.y + a * accum10.x;
    
    tmp3_re = accum10.x - a * accum4.y;
    tmp3_im = accum10.y + a * accum4.x;
    
    o01_re = b * o01_re + tmp0_re;
    o01_im = b * o01_im + tmp0_im;
    o11_re = b * o11_re + tmp1_re;
    o11_im = b * o11_im + tmp1_im;
    o21_re = b * o21_re + tmp2_re;
    o21_im = b * o21_im + tmp2_im;
    o31_re = b * o31_re + tmp3_re;
    o31_im = b * o31_im + tmp3_im;
    
    //get the 3d color component:    
    
    tmp0_re = accum2.x - a * accum8.y;
    tmp0_im = accum2.y + a * accum8.x;
    
    tmp2_re = accum8.x - a * accum2.y;
    tmp2_im = accum8.y + a * accum2.x;

    tmp1_re = accum5.x - a * accum11.y;
    tmp1_im = accum5.y + a * accum11.x;
    
    tmp3_re = accum11.x - a * accum5.y;
    tmp3_im = accum11.y + a * accum5.x;
    
    o02_re = b * o02_re + tmp0_re;//2
    o02_im = b * o02_im + tmp0_im;//2
    o12_re = b * o12_re + tmp1_re;//5
    o12_im = b * o12_im + tmp1_im;//5
    o22_re = b * o22_re + tmp2_re;//8
    o22_im = b * o22_im + tmp2_im;//8
    o32_re = b * o32_re + tmp3_re;//11
    o32_im = b * o32_im + tmp3_im;//11

#else

#define tmp0_re tmp0.x
#define tmp0_im tmp0.y
#define tmp1_re tmp0.z
#define tmp1_im tmp0.w
#define tmp2_re tmp1.x
#define tmp2_im tmp1.y
#define tmp3_re tmp1.z
#define tmp3_im tmp1.w

    float4 tmp0, tmp1;
    
    //apply (1 + i*a*gamma_5) to the input spinor and then add to (b * output spinor)
    
    //get the 1st color component:(accum0.xy, accum1.zw, accum3.xy, accum4.zw)

    tmp0_re = accum0.x - a * accum3.y;
    tmp0_im = accum0.y + a * accum3.x;
    
    tmp1_re = accum1.z - a * accum4.w;
    tmp1_im = accum1.w + a * accum4.z;
    
    tmp2_re = accum3.x - a * accum0.y;
    tmp2_im = accum3.y + a * accum0.x;

    tmp3_re = accum4.z - a * accum1.w;
    tmp3_im = accum4.w + a * accum1.z;
    
    o00_re = b * o00_re + tmp0_re;
    o00_im = b * o00_im + tmp0_im;
    o10_re = b * o10_re + tmp1_re;
    o10_im = b * o10_im + tmp1_im;
    o20_re = b * o20_re + tmp2_re;
    o20_im = b * o20_im + tmp2_im;
    o30_re = b * o30_re + tmp3_re;
    o30_im = b * o30_im + tmp3_im;

    //get the 2nd color component:(accum0.zw, accum2.xy, accum3.zw, accum5.xy)

    tmp0_re = accum0.z - a * accum3.w;
    tmp0_im = accum0.w + a * accum3.z;
    
    tmp1_re = accum2.x - a * accum5.y;
    tmp1_im = accum2.y + a * accum5.x;
    
    tmp2_re = accum3.z - a * accum0.w;
    tmp2_im = accum3.w + a * accum0.z;

    tmp3_re = accum5.x - a * accum2.y;
    tmp3_im = accum5.y + a * accum2.x;
    
    o01_re = b * o01_re + tmp0_re;
    o01_im = b * o01_im + tmp0_im;
    o11_re = b * o11_re + tmp1_re;
    o11_im = b * o11_im + tmp1_im;
    o21_re = b * o21_re + tmp2_re;
    o21_im = b * o21_im + tmp2_im;
    o31_re = b * o31_re + tmp3_re;
    o31_im = b * o31_im + tmp3_im;
    
    //get the 3d color component:(accum1.xy, accum2.zw, accum4.xy, accum5.zw)

    tmp0_re = accum1.x - a * accum4.y;
    tmp0_im = accum1.y + a * accum4.x;
    
    tmp1_re = accum2.z - a * accum5.w;
    tmp1_im = accum2.w + a * accum5.z;
    
    tmp2_re = accum4.x - a * accum1.y;
    tmp2_im = accum4.y + a * accum1.x;

    tmp3_re = accum5.z - a * accum2.w;
    tmp3_im = accum5.w + a * accum2.z;
    
    o02_re = b * o02_re + tmp0_re;
    o02_im = b * o02_im + tmp0_im;
    o12_re = b * o12_re + tmp1_re;
    o12_im = b * o12_im + tmp1_im;
    o22_re = b * o22_re + tmp2_re;
    o22_im = b * o22_im + tmp2_im;
    o32_re = b * o32_re + tmp3_re;
    o32_im = b * o32_im + tmp3_im;
    

#endif // SPINOR_DOUBLE

#undef tmp0_re
#undef tmp0_im
#undef tmp1_re
#undef tmp1_im
#undef tmp2_re
#undef tmp2_im
#undef tmp3_re
#undef tmp3_im

#endif // DSLASH_XPAY

//Note: in total 48 + 4 = 52 additional fp operations for dslashXpay kernels 

    // write spinor field back to device memory
    WRITE_SPINOR();

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

#undef A_re
#undef A_im

#undef g00_re
#undef g00_im
#undef g01_re
#undef g01_im
#undef g02_re
#undef g02_im
#undef g10_re
#undef g10_im
#undef g11_re
#undef g11_im
#undef g12_re
#undef g12_im
#undef g20_re
#undef g20_im
#undef g21_re
#undef g21_im
#undef g22_re
#undef g22_im

#undef i00_re
#undef i00_im
#undef i01_re
#undef i01_im
#undef i02_re
#undef i02_im
#undef i10_re
#undef i10_im
#undef i11_re
#undef i11_im
#undef i12_re
#undef i12_im
#undef i20_re
#undef i20_im
#undef i21_re
#undef i21_im
#undef i22_re
#undef i22_im
#undef i30_re
#undef i30_im
#undef i31_re
#undef i31_im
#undef i32_re
#undef i32_im

