// *** CUDA DSLASH ***

//Extra constants (double) mu, (double)eta and (double)delta

#define SHARED_TMNDEG_FLOATS_PER_THREAD 0

// NB! Don't trust any MULTI_GPU code

#if (CUDA_VERSION >= 4010)
#define VOLATILE
#else
#define VOLATILE volatile
#endif
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
#ifdef GAUGE_FLOAT2
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
VOLATILE spinorFloat o1_00_re;
VOLATILE spinorFloat o1_00_im;
VOLATILE spinorFloat o1_01_re;
VOLATILE spinorFloat o1_01_im;
VOLATILE spinorFloat o1_02_re;
VOLATILE spinorFloat o1_02_im;
VOLATILE spinorFloat o1_10_re;
VOLATILE spinorFloat o1_10_im;
VOLATILE spinorFloat o1_11_re;
VOLATILE spinorFloat o1_11_im;
VOLATILE spinorFloat o1_12_re;
VOLATILE spinorFloat o1_12_im;
VOLATILE spinorFloat o1_20_re;
VOLATILE spinorFloat o1_20_im;
VOLATILE spinorFloat o1_21_re;
VOLATILE spinorFloat o1_21_im;
VOLATILE spinorFloat o1_22_re;
VOLATILE spinorFloat o1_22_im;
VOLATILE spinorFloat o1_30_re;
VOLATILE spinorFloat o1_30_im;
VOLATILE spinorFloat o1_31_re;
VOLATILE spinorFloat o1_31_im;
VOLATILE spinorFloat o1_32_re;
VOLATILE spinorFloat o1_32_im;

VOLATILE spinorFloat o2_00_re;
VOLATILE spinorFloat o2_00_im;
VOLATILE spinorFloat o2_01_re;
VOLATILE spinorFloat o2_01_im;
VOLATILE spinorFloat o2_02_re;
VOLATILE spinorFloat o2_02_im;
VOLATILE spinorFloat o2_10_re;
VOLATILE spinorFloat o2_10_im;
VOLATILE spinorFloat o2_11_re;
VOLATILE spinorFloat o2_11_im;
VOLATILE spinorFloat o2_12_re;
VOLATILE spinorFloat o2_12_im;
VOLATILE spinorFloat o2_20_re;
VOLATILE spinorFloat o2_20_im;
VOLATILE spinorFloat o2_21_re;
VOLATILE spinorFloat o2_21_im;
VOLATILE spinorFloat o2_22_re;
VOLATILE spinorFloat o2_22_im;
VOLATILE spinorFloat o2_30_re;
VOLATILE spinorFloat o2_30_im;
VOLATILE spinorFloat o2_31_re;
VOLATILE spinorFloat o2_31_im;
VOLATILE spinorFloat o2_32_re;
VOLATILE spinorFloat o2_32_im;

#ifdef SPINOR_DOUBLE
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 8 // to avoid bank conflicts on G80 and GT200
#endif
#else
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
#endif

#include "read_gauge.h"
#include "io_spinor.h"

int x1, x2, x3, x4;
int X;

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

X = 2*sid;
int aux1 = X / X1;
x1 = X - aux1 * X1;
int aux2 = aux1 / X2;
x2 = aux1 - aux2 * X2;
x4 = aux2 / X3;
x3 = aux2 - x4 * X3;
aux1 = (param.parity + x4 + x3 + x2) & 1;
x1 += aux1;
X += aux1;

o1_00_re = o1_00_im = 0;
o1_01_re = o1_01_im = 0;
o1_02_re = o1_02_im = 0;
o1_10_re = o1_10_im = 0;
o1_11_re = o1_11_im = 0;
o1_12_re = o1_12_im = 0;
o1_20_re = o1_20_im = 0;
o1_21_re = o1_21_im = 0;
o1_22_re = o1_22_im = 0;
o1_30_re = o1_30_im = 0;
o1_31_re = o1_31_im = 0;
o1_32_re = o1_32_im = 0;

o2_00_re = o2_00_im = 0;
o2_01_re = o2_01_im = 0;
o2_02_re = o2_02_im = 0;
o2_10_re = o2_10_im = 0;
o2_11_re = o2_11_im = 0;
o2_12_re = o2_12_im = 0;
o2_20_re = o2_20_im = 0;
o2_21_re = o2_21_im = 0;
o2_22_re = o2_22_im = 0;
o2_30_re = o2_30_im = 0;
o2_31_re = o2_31_im = 0;
o2_32_re = o2_32_im = 0;

{
    // Projector P0-
    // 1 0 0 -i 
    // 0 1 -i 0 
    // 0 i 1 0 
    // i 0 0 1 
    
    int sp_idx = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 0, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(0);
    
    {// read the first flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
    
    	spinorFloat A0_re = 0;
    	A0_re += g00_re * a0_re;
    	A0_re -= g00_im * a0_im;
    	A0_re += g01_re * a1_re;
    	A0_re -= g01_im * a1_im;
    	A0_re += g02_re * a2_re;
    	A0_re -= g02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += g00_re * a0_im;
    	A0_im += g00_im * a0_re;
    	A0_im += g01_re * a1_im;
    	A0_im += g01_im * a1_re;
    	A0_im += g02_re * a2_im;
    	A0_im += g02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += g00_re * b0_re;
    	B0_re -= g00_im * b0_im;
    	B0_re += g01_re * b1_re;
    	B0_re -= g01_im * b1_im;
    	B0_re += g02_re * b2_re;
    	B0_re -= g02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += g00_re * b0_im;
    	B0_im += g00_im * b0_re;
    	B0_im += g01_re * b1_im;
    	B0_im += g01_im * b1_re;
    	B0_im += g02_re * b2_im;
    	B0_im += g02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += g10_re * a0_re;
    	A1_re -= g10_im * a0_im;
    	A1_re += g11_re * a1_re;
    	A1_re -= g11_im * a1_im;
    	A1_re += g12_re * a2_re;
    	A1_re -= g12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += g10_re * a0_im;
    	A1_im += g10_im * a0_re;
    	A1_im += g11_re * a1_im;
    	A1_im += g11_im * a1_re;
    	A1_im += g12_re * a2_im;
    	A1_im += g12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += g10_re * b0_re;
    	B1_re -= g10_im * b0_im;
    	B1_re += g11_re * b1_re;
    	B1_re -= g11_im * b1_im;
    	B1_re += g12_re * b2_re;
    	B1_re -= g12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += g10_re * b0_im;
    	B1_im += g10_im * b0_re;
    	B1_im += g11_re * b1_im;
    	B1_im += g11_im * b1_re;
    	B1_im += g12_re * b2_im;
    	B1_im += g12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += g20_re * a0_re;
    	A2_re -= g20_im * a0_im;
    	A2_re += g21_re * a1_re;
    	A2_re -= g21_im * a1_im;
    	A2_re += g22_re * a2_re;
    	A2_re -= g22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += g20_re * a0_im;
    	A2_im += g20_im * a0_re;
    	A2_im += g21_re * a1_im;
    	A2_im += g21_im * a1_re;
    	A2_im += g22_re * a2_im;
    	A2_im += g22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += g20_re * b0_re;
    	B2_re -= g20_im * b0_im;
    	B2_re += g21_re * b1_re;
    	B2_re -= g21_im * b1_im;
    	B2_re += g22_re * b2_re;
    	B2_re -= g22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += g20_re * b0_im;
    	B2_im += g20_im * b0_re;
    	B2_im += g21_re * b1_im;
    	B2_im += g21_im * b1_re;
    	B2_im += g22_re * b2_im;
    	B2_im += g22_im * b2_re;
	

    	o1_00_re += A0_re;
    	o1_00_im += A0_im;
    	o1_10_re += B0_re;
    	o1_10_im += B0_im;
    	o1_20_re -= B0_im;
    	o1_20_im += B0_re;
    	o1_30_re -= A0_im;
    	o1_30_im += A0_re;
    	
    	o1_01_re += A1_re;
    	o1_01_im += A1_im;
    	o1_11_re += B1_re;
    	o1_11_im += B1_im;
    	o1_21_re -= B1_im;
    	o1_21_im += B1_re;
    	o1_31_re -= A1_im;
    	o1_31_im += A1_re;
    	
    	o1_02_re += A2_re;
    	o1_02_im += A2_im;
    	o1_12_re += B2_re;
    	o1_12_im += B2_im;
    	o1_22_re -= B2_im;
    	o1_22_im += B2_re;
    	o1_32_re -= A2_im;
    	o1_32_im += A2_re;
	
    }
    {// read the second flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
        
    	spinorFloat A0_re = 0;
    	A0_re += g00_re * a0_re;
    	A0_re -= g00_im * a0_im;
    	A0_re += g01_re * a1_re;
    	A0_re -= g01_im * a1_im;
    	A0_re += g02_re * a2_re;
    	A0_re -= g02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += g00_re * a0_im;
    	A0_im += g00_im * a0_re;
    	A0_im += g01_re * a1_im;
    	A0_im += g01_im * a1_re;
    	A0_im += g02_re * a2_im;
    	A0_im += g02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += g00_re * b0_re;
    	B0_re -= g00_im * b0_im;
    	B0_re += g01_re * b1_re;
    	B0_re -= g01_im * b1_im;
    	B0_re += g02_re * b2_re;
    	B0_re -= g02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += g00_re * b0_im;
    	B0_im += g00_im * b0_re;
    	B0_im += g01_re * b1_im;
    	B0_im += g01_im * b1_re;
    	B0_im += g02_re * b2_im;
    	B0_im += g02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += g10_re * a0_re;
    	A1_re -= g10_im * a0_im;
    	A1_re += g11_re * a1_re;
    	A1_re -= g11_im * a1_im;
    	A1_re += g12_re * a2_re;
    	A1_re -= g12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += g10_re * a0_im;
    	A1_im += g10_im * a0_re;
    	A1_im += g11_re * a1_im;
    	A1_im += g11_im * a1_re;
    	A1_im += g12_re * a2_im;
    	A1_im += g12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += g10_re * b0_re;
    	B1_re -= g10_im * b0_im;
    	B1_re += g11_re * b1_re;
    	B1_re -= g11_im * b1_im;
    	B1_re += g12_re * b2_re;
    	B1_re -= g12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += g10_re * b0_im;
    	B1_im += g10_im * b0_re;
    	B1_im += g11_re * b1_im;
    	B1_im += g11_im * b1_re;
    	B1_im += g12_re * b2_im;
    	B1_im += g12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += g20_re * a0_re;
    	A2_re -= g20_im * a0_im;
    	A2_re += g21_re * a1_re;
    	A2_re -= g21_im * a1_im;
    	A2_re += g22_re * a2_re;
    	A2_re -= g22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += g20_re * a0_im;
    	A2_im += g20_im * a0_re;
    	A2_im += g21_re * a1_im;
    	A2_im += g21_im * a1_re;
    	A2_im += g22_re * a2_im;
    	A2_im += g22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += g20_re * b0_re;
    	B2_re -= g20_im * b0_im;
    	B2_re += g21_re * b1_re;
    	B2_re -= g21_im * b1_im;
    	B2_re += g22_re * b2_re;
    	B2_re -= g22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += g20_re * b0_im;
    	B2_im += g20_im * b0_re;
    	B2_im += g21_re * b1_im;
    	B2_im += g21_im * b1_re;
    	B2_im += g22_re * b2_im;
    	B2_im += g22_im * b2_re;
	

    	o2_00_re += A0_re;
    	o2_00_im += A0_im;
    	o2_10_re += B0_re;
    	o2_10_im += B0_im;
    	o2_20_re -= B0_im;
    	o2_20_im += B0_re;
    	o2_30_re -= A0_im;
    	o2_30_im += A0_re;
    	
    	o2_01_re += A1_re;
    	o2_01_im += A1_im;
    	o2_11_re += B1_re;
    	o2_11_im += B1_im;
    	o2_21_re -= B1_im;
    	o2_21_im += B1_re;
    	o2_31_re -= A1_im;
    	o2_31_im += A1_re;
    	
    	o2_02_re += A2_re;
    	o2_02_im += A2_im;
    	o2_12_re += B2_re;
    	o2_12_im += B2_im;
    	o2_22_re -= B2_im;
    	o2_22_im += B2_re;
    	o2_32_re -= A2_im;
    	o2_32_im += A2_re;

    }
}

{
    // Projector P0+
    // 1 0 0 i 
    // 0 1 i 0 
    // 0 -i 1 0 
    // -i 0 0 1 
    
    int sp_idx = ((x1==0)    ? X+X1m1 : X-1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 1, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(1);
    
    {// read the first flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += gT00_re * a0_re;
    	A0_re -= gT00_im * a0_im;
    	A0_re += gT01_re * a1_re;
    	A0_re -= gT01_im * a1_im;
    	A0_re += gT02_re * a2_re;
    	A0_re -= gT02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += gT00_re * a0_im;
    	A0_im += gT00_im * a0_re;
    	A0_im += gT01_re * a1_im;
    	A0_im += gT01_im * a1_re;
    	A0_im += gT02_re * a2_im;
    	A0_im += gT02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += gT00_re * b0_re;
    	B0_re -= gT00_im * b0_im;
    	B0_re += gT01_re * b1_re;
    	B0_re -= gT01_im * b1_im;
    	B0_re += gT02_re * b2_re;
    	B0_re -= gT02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += gT00_re * b0_im;
    	B0_im += gT00_im * b0_re;
    	B0_im += gT01_re * b1_im;
    	B0_im += gT01_im * b1_re;
    	B0_im += gT02_re * b2_im;
    	B0_im += gT02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += gT10_re * a0_re;
    	A1_re -= gT10_im * a0_im;
    	A1_re += gT11_re * a1_re;
    	A1_re -= gT11_im * a1_im;
    	A1_re += gT12_re * a2_re;
    	A1_re -= gT12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += gT10_re * a0_im;
    	A1_im += gT10_im * a0_re;
    	A1_im += gT11_re * a1_im;
    	A1_im += gT11_im * a1_re;
    	A1_im += gT12_re * a2_im;
    	A1_im += gT12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += gT10_re * b0_re;
    	B1_re -= gT10_im * b0_im;
    	B1_re += gT11_re * b1_re;
    	B1_re -= gT11_im * b1_im;
    	B1_re += gT12_re * b2_re;
    	B1_re -= gT12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += gT10_re * b0_im;
    	B1_im += gT10_im * b0_re;
    	B1_im += gT11_re * b1_im;
    	B1_im += gT11_im * b1_re;
    	B1_im += gT12_re * b2_im;
    	B1_im += gT12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += gT20_re * a0_re;
    	A2_re -= gT20_im * a0_im;
    	A2_re += gT21_re * a1_re;
    	A2_re -= gT21_im * a1_im;
    	A2_re += gT22_re * a2_re;
    	A2_re -= gT22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += gT20_re * a0_im;
    	A2_im += gT20_im * a0_re;
    	A2_im += gT21_re * a1_im;
    	A2_im += gT21_im * a1_re;
    	A2_im += gT22_re * a2_im;
    	A2_im += gT22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += gT20_re * b0_re;
    	B2_re -= gT20_im * b0_im;
    	B2_re += gT21_re * b1_re;
    	B2_re -= gT21_im * b1_im;
    	B2_re += gT22_re * b2_re;
    	B2_re -= gT22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += gT20_re * b0_im;
    	B2_im += gT20_im * b0_re;
    	B2_im += gT21_re * b1_im;
    	B2_im += gT21_im * b1_re;
    	B2_im += gT22_re * b2_im;
    	B2_im += gT22_im * b2_re;
    	
    
    	o1_00_re += A0_re;
    	o1_00_im += A0_im;
    	o1_10_re += B0_re;
    	o1_10_im += B0_im;
    	o1_20_re += B0_im;
    	o1_20_im -= B0_re;
    	o1_30_re += A0_im;
    	o1_30_im -= A0_re;
    	
    	o1_01_re += A1_re;
    	o1_01_im += A1_im;
    	o1_11_re += B1_re;
    	o1_11_im += B1_im;
    	o1_21_re += B1_im;
    	o1_21_im -= B1_re;
    	o1_31_re += A1_im;
    	o1_31_im -= A1_re;
    	
    	o1_02_re += A2_re;
    	o1_02_im += A2_im;
    	o1_12_re += B2_re;
    	o1_12_im += B2_im;
    	o1_22_re += B2_im;
    	o1_22_im -= B2_re;
    	o1_32_re += A2_im;
    	o1_32_im -= A2_re;
    	
    }
    {// read the second flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += gT00_re * a0_re;
    	A0_re -= gT00_im * a0_im;
    	A0_re += gT01_re * a1_re;
    	A0_re -= gT01_im * a1_im;
    	A0_re += gT02_re * a2_re;
    	A0_re -= gT02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += gT00_re * a0_im;
    	A0_im += gT00_im * a0_re;
    	A0_im += gT01_re * a1_im;
    	A0_im += gT01_im * a1_re;
    	A0_im += gT02_re * a2_im;
    	A0_im += gT02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += gT00_re * b0_re;
    	B0_re -= gT00_im * b0_im;
    	B0_re += gT01_re * b1_re;
    	B0_re -= gT01_im * b1_im;
    	B0_re += gT02_re * b2_re;
    	B0_re -= gT02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += gT00_re * b0_im;
    	B0_im += gT00_im * b0_re;
    	B0_im += gT01_re * b1_im;
    	B0_im += gT01_im * b1_re;
    	B0_im += gT02_re * b2_im;
    	B0_im += gT02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += gT10_re * a0_re;
    	A1_re -= gT10_im * a0_im;
    	A1_re += gT11_re * a1_re;
    	A1_re -= gT11_im * a1_im;
    	A1_re += gT12_re * a2_re;
    	A1_re -= gT12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += gT10_re * a0_im;
    	A1_im += gT10_im * a0_re;
    	A1_im += gT11_re * a1_im;
    	A1_im += gT11_im * a1_re;
    	A1_im += gT12_re * a2_im;
    	A1_im += gT12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += gT10_re * b0_re;
    	B1_re -= gT10_im * b0_im;
    	B1_re += gT11_re * b1_re;
    	B1_re -= gT11_im * b1_im;
    	B1_re += gT12_re * b2_re;
    	B1_re -= gT12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += gT10_re * b0_im;
    	B1_im += gT10_im * b0_re;
    	B1_im += gT11_re * b1_im;
    	B1_im += gT11_im * b1_re;
    	B1_im += gT12_re * b2_im;
    	B1_im += gT12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += gT20_re * a0_re;
    	A2_re -= gT20_im * a0_im;
    	A2_re += gT21_re * a1_re;
    	A2_re -= gT21_im * a1_im;
    	A2_re += gT22_re * a2_re;
    	A2_re -= gT22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += gT20_re * a0_im;
    	A2_im += gT20_im * a0_re;
    	A2_im += gT21_re * a1_im;
    	A2_im += gT21_im * a1_re;
    	A2_im += gT22_re * a2_im;
    	A2_im += gT22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += gT20_re * b0_re;
    	B2_re -= gT20_im * b0_im;
    	B2_re += gT21_re * b1_re;
    	B2_re -= gT21_im * b1_im;
    	B2_re += gT22_re * b2_re;
    	B2_re -= gT22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += gT20_re * b0_im;
    	B2_im += gT20_im * b0_re;
    	B2_im += gT21_re * b1_im;
    	B2_im += gT21_im * b1_re;
    	B2_im += gT22_re * b2_im;
    	B2_im += gT22_im * b2_re;
    	
    
    	o2_00_re += A0_re;
    	o2_00_im += A0_im;
    	o2_10_re += B0_re;
    	o2_10_im += B0_im;
    	o2_20_re += B0_im;
    	o2_20_im -= B0_re;
    	o2_30_re += A0_im;
    	o2_30_im -= A0_re;
    	
    	o2_01_re += A1_re;
    	o2_01_im += A1_im;
    	o2_11_re += B1_re;
    	o2_11_im += B1_im;
    	o2_21_re += B1_im;
    	o2_21_im -= B1_re;
    	o2_31_re += A1_im;
    	o2_31_im -= A1_re;
    	
    	o2_02_re += A2_re;
    	o2_02_im += A2_im;
    	o2_12_re += B2_re;
    	o2_12_im += B2_im;
    	o2_22_re += B2_im;
    	o2_22_im -= B2_re;
    	o2_32_re += A2_im;
    	o2_32_im -= A2_re;
    	
    }
}
{
    // Projector P1-
    // 1 0 0 -1 
    // 0 1 1 0 
    // 0 1 1 0 
    // -1 0 0 1 
    
    int sp_idx = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 2, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(2);
    
    {// read the first flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += g00_re * a0_re;
    	A0_re -= g00_im * a0_im;
    	A0_re += g01_re * a1_re;
    	A0_re -= g01_im * a1_im;
    	A0_re += g02_re * a2_re;
    	A0_re -= g02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += g00_re * a0_im;
    	A0_im += g00_im * a0_re;
    	A0_im += g01_re * a1_im;
    	A0_im += g01_im * a1_re;
    	A0_im += g02_re * a2_im;
    	A0_im += g02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += g00_re * b0_re;
    	B0_re -= g00_im * b0_im;
    	B0_re += g01_re * b1_re;
    	B0_re -= g01_im * b1_im;
    	B0_re += g02_re * b2_re;
    	B0_re -= g02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += g00_re * b0_im;
    	B0_im += g00_im * b0_re;
    	B0_im += g01_re * b1_im;
    	B0_im += g01_im * b1_re;
    	B0_im += g02_re * b2_im;
    	B0_im += g02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += g10_re * a0_re;
    	A1_re -= g10_im * a0_im;
    	A1_re += g11_re * a1_re;
    	A1_re -= g11_im * a1_im;
    	A1_re += g12_re * a2_re;
    	A1_re -= g12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += g10_re * a0_im;
    	A1_im += g10_im * a0_re;
    	A1_im += g11_re * a1_im;
    	A1_im += g11_im * a1_re;
    	A1_im += g12_re * a2_im;
    	A1_im += g12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += g10_re * b0_re;
    	B1_re -= g10_im * b0_im;
    	B1_re += g11_re * b1_re;
    	B1_re -= g11_im * b1_im;
    	B1_re += g12_re * b2_re;
    	B1_re -= g12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += g10_re * b0_im;
    	B1_im += g10_im * b0_re;
    	B1_im += g11_re * b1_im;
    	B1_im += g11_im * b1_re;
    	B1_im += g12_re * b2_im;
    	B1_im += g12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += g20_re * a0_re;
    	A2_re -= g20_im * a0_im;
    	A2_re += g21_re * a1_re;
    	A2_re -= g21_im * a1_im;
    	A2_re += g22_re * a2_re;
    	A2_re -= g22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += g20_re * a0_im;
    	A2_im += g20_im * a0_re;
    	A2_im += g21_re * a1_im;
    	A2_im += g21_im * a1_re;
    	A2_im += g22_re * a2_im;
    	A2_im += g22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += g20_re * b0_re;
    	B2_re -= g20_im * b0_im;
    	B2_re += g21_re * b1_re;
    	B2_re -= g21_im * b1_im;
    	B2_re += g22_re * b2_re;
    	B2_re -= g22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += g20_re * b0_im;
    	B2_im += g20_im * b0_re;
    	B2_im += g21_re * b1_im;
    	B2_im += g21_im * b1_re;
    	B2_im += g22_re * b2_im;
    	B2_im += g22_im * b2_re;
    	
    
    	o1_00_re += A0_re;
    	o1_00_im += A0_im;
    	o1_10_re += B0_re;
    	o1_10_im += B0_im;
    	o1_20_re += B0_re;
    	o1_20_im += B0_im;
    	o1_30_re -= A0_re;
    	o1_30_im -= A0_im;
    	
    	o1_01_re += A1_re;
    	o1_01_im += A1_im;
    	o1_11_re += B1_re;
    	o1_11_im += B1_im;
    	o1_21_re += B1_re;
    	o1_21_im += B1_im;
    	o1_31_re -= A1_re;
    	o1_31_im -= A1_im;
    	
    	o1_02_re += A2_re;
    	o1_02_im += A2_im;
    	o1_12_re += B2_re;
    	o1_12_im += B2_im;
    	o1_22_re += B2_re;
    	o1_22_im += B2_im;
    	o1_32_re -= A2_re;
    	o1_32_im -= A2_im;
    	
    }
    {// read the second flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += g00_re * a0_re;
    	A0_re -= g00_im * a0_im;
    	A0_re += g01_re * a1_re;
    	A0_re -= g01_im * a1_im;
    	A0_re += g02_re * a2_re;
    	A0_re -= g02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += g00_re * a0_im;
    	A0_im += g00_im * a0_re;
    	A0_im += g01_re * a1_im;
    	A0_im += g01_im * a1_re;
    	A0_im += g02_re * a2_im;
    	A0_im += g02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += g00_re * b0_re;
    	B0_re -= g00_im * b0_im;
    	B0_re += g01_re * b1_re;
    	B0_re -= g01_im * b1_im;
    	B0_re += g02_re * b2_re;
    	B0_re -= g02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += g00_re * b0_im;
    	B0_im += g00_im * b0_re;
    	B0_im += g01_re * b1_im;
    	B0_im += g01_im * b1_re;
    	B0_im += g02_re * b2_im;
    	B0_im += g02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += g10_re * a0_re;
    	A1_re -= g10_im * a0_im;
    	A1_re += g11_re * a1_re;
    	A1_re -= g11_im * a1_im;
    	A1_re += g12_re * a2_re;
    	A1_re -= g12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += g10_re * a0_im;
    	A1_im += g10_im * a0_re;
    	A1_im += g11_re * a1_im;
    	A1_im += g11_im * a1_re;
    	A1_im += g12_re * a2_im;
    	A1_im += g12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += g10_re * b0_re;
    	B1_re -= g10_im * b0_im;
    	B1_re += g11_re * b1_re;
    	B1_re -= g11_im * b1_im;
    	B1_re += g12_re * b2_re;
    	B1_re -= g12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += g10_re * b0_im;
    	B1_im += g10_im * b0_re;
    	B1_im += g11_re * b1_im;
    	B1_im += g11_im * b1_re;
    	B1_im += g12_re * b2_im;
    	B1_im += g12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += g20_re * a0_re;
    	A2_re -= g20_im * a0_im;
    	A2_re += g21_re * a1_re;
    	A2_re -= g21_im * a1_im;
    	A2_re += g22_re * a2_re;
    	A2_re -= g22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += g20_re * a0_im;
    	A2_im += g20_im * a0_re;
    	A2_im += g21_re * a1_im;
    	A2_im += g21_im * a1_re;
    	A2_im += g22_re * a2_im;
    	A2_im += g22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += g20_re * b0_re;
    	B2_re -= g20_im * b0_im;
    	B2_re += g21_re * b1_re;
    	B2_re -= g21_im * b1_im;
    	B2_re += g22_re * b2_re;
    	B2_re -= g22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += g20_re * b0_im;
    	B2_im += g20_im * b0_re;
    	B2_im += g21_re * b1_im;
    	B2_im += g21_im * b1_re;
    	B2_im += g22_re * b2_im;
    	B2_im += g22_im * b2_re;
    	
    
    	o2_00_re += A0_re;
    	o2_00_im += A0_im;
    	o2_10_re += B0_re;
    	o2_10_im += B0_im;
    	o2_20_re += B0_re;
    	o2_20_im += B0_im;
    	o2_30_re -= A0_re;
    	o2_30_im -= A0_im;
    	
    	o2_01_re += A1_re;
    	o2_01_im += A1_im;
    	o2_11_re += B1_re;
    	o2_11_im += B1_im;
    	o2_21_re += B1_re;
    	o2_21_im += B1_im;
    	o2_31_re -= A1_re;
    	o2_31_im -= A1_im;
    	
    	o2_02_re += A2_re;
    	o2_02_im += A2_im;
    	o2_12_re += B2_re;
    	o2_12_im += B2_im;
    	o2_22_re += B2_re;
    	o2_22_im += B2_im;
    	o2_32_re -= A2_re;
    	o2_32_im -= A2_im;
    	
    }
}

{
    // Projector P1+
    // 1 0 0 1 
    // 0 1 -1 0 
    // 0 -1 1 0 
    // 1 0 0 1 
    
    int sp_idx = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 3, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(3);
    
    {// read the first flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += gT00_re * a0_re;
    	A0_re -= gT00_im * a0_im;
    	A0_re += gT01_re * a1_re;
    	A0_re -= gT01_im * a1_im;
    	A0_re += gT02_re * a2_re;
    	A0_re -= gT02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += gT00_re * a0_im;
    	A0_im += gT00_im * a0_re;
    	A0_im += gT01_re * a1_im;
    	A0_im += gT01_im * a1_re;
    	A0_im += gT02_re * a2_im;
    	A0_im += gT02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += gT00_re * b0_re;
    	B0_re -= gT00_im * b0_im;
    	B0_re += gT01_re * b1_re;
    	B0_re -= gT01_im * b1_im;
    	B0_re += gT02_re * b2_re;
    	B0_re -= gT02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += gT00_re * b0_im;
    	B0_im += gT00_im * b0_re;
    	B0_im += gT01_re * b1_im;
    	B0_im += gT01_im * b1_re;
    	B0_im += gT02_re * b2_im;
    	B0_im += gT02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += gT10_re * a0_re;
    	A1_re -= gT10_im * a0_im;
    	A1_re += gT11_re * a1_re;
    	A1_re -= gT11_im * a1_im;
    	A1_re += gT12_re * a2_re;
    	A1_re -= gT12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += gT10_re * a0_im;
    	A1_im += gT10_im * a0_re;
    	A1_im += gT11_re * a1_im;
    	A1_im += gT11_im * a1_re;
    	A1_im += gT12_re * a2_im;
    	A1_im += gT12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += gT10_re * b0_re;
    	B1_re -= gT10_im * b0_im;
    	B1_re += gT11_re * b1_re;
    	B1_re -= gT11_im * b1_im;
    	B1_re += gT12_re * b2_re;
    	B1_re -= gT12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += gT10_re * b0_im;
    	B1_im += gT10_im * b0_re;
    	B1_im += gT11_re * b1_im;
    	B1_im += gT11_im * b1_re;
    	B1_im += gT12_re * b2_im;
    	B1_im += gT12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += gT20_re * a0_re;
    	A2_re -= gT20_im * a0_im;
    	A2_re += gT21_re * a1_re;
    	A2_re -= gT21_im * a1_im;
    	A2_re += gT22_re * a2_re;
    	A2_re -= gT22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += gT20_re * a0_im;
    	A2_im += gT20_im * a0_re;
    	A2_im += gT21_re * a1_im;
    	A2_im += gT21_im * a1_re;
    	A2_im += gT22_re * a2_im;
    	A2_im += gT22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += gT20_re * b0_re;
    	B2_re -= gT20_im * b0_im;
    	B2_re += gT21_re * b1_re;
    	B2_re -= gT21_im * b1_im;
    	B2_re += gT22_re * b2_re;
    	B2_re -= gT22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += gT20_re * b0_im;
    	B2_im += gT20_im * b0_re;
    	B2_im += gT21_re * b1_im;
    	B2_im += gT21_im * b1_re;
    	B2_im += gT22_re * b2_im;
    	B2_im += gT22_im * b2_re;
    	
    
    	o1_00_re += A0_re;
    	o1_00_im += A0_im;
    	o1_10_re += B0_re;
    	o1_10_im += B0_im;
    	o1_20_re -= B0_re;
    	o1_20_im -= B0_im;
    	o1_30_re += A0_re;
    	o1_30_im += A0_im;
    	
    	o1_01_re += A1_re;
    	o1_01_im += A1_im;
    	o1_11_re += B1_re;
    	o1_11_im += B1_im;
    	o1_21_re -= B1_re;
    	o1_21_im -= B1_im;
    	o1_31_re += A1_re;
    	o1_31_im += A1_im;
    	
    	o1_02_re += A2_re;
    	o1_02_im += A2_im;
    	o1_12_re += B2_re;
    	o1_12_im += B2_im;
    	o1_22_re -= B2_re;
    	o1_22_im -= B2_im;
    	o1_32_re += A2_re;
    	o1_32_im += A2_im;
    	
    }
    {// read the second flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += gT00_re * a0_re;
    	A0_re -= gT00_im * a0_im;
    	A0_re += gT01_re * a1_re;
    	A0_re -= gT01_im * a1_im;
    	A0_re += gT02_re * a2_re;
    	A0_re -= gT02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += gT00_re * a0_im;
    	A0_im += gT00_im * a0_re;
    	A0_im += gT01_re * a1_im;
    	A0_im += gT01_im * a1_re;
    	A0_im += gT02_re * a2_im;
    	A0_im += gT02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += gT00_re * b0_re;
    	B0_re -= gT00_im * b0_im;
    	B0_re += gT01_re * b1_re;
    	B0_re -= gT01_im * b1_im;
    	B0_re += gT02_re * b2_re;
    	B0_re -= gT02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += gT00_re * b0_im;
    	B0_im += gT00_im * b0_re;
    	B0_im += gT01_re * b1_im;
    	B0_im += gT01_im * b1_re;
    	B0_im += gT02_re * b2_im;
    	B0_im += gT02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += gT10_re * a0_re;
    	A1_re -= gT10_im * a0_im;
    	A1_re += gT11_re * a1_re;
    	A1_re -= gT11_im * a1_im;
    	A1_re += gT12_re * a2_re;
    	A1_re -= gT12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += gT10_re * a0_im;
    	A1_im += gT10_im * a0_re;
    	A1_im += gT11_re * a1_im;
    	A1_im += gT11_im * a1_re;
    	A1_im += gT12_re * a2_im;
    	A1_im += gT12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += gT10_re * b0_re;
    	B1_re -= gT10_im * b0_im;
    	B1_re += gT11_re * b1_re;
    	B1_re -= gT11_im * b1_im;
    	B1_re += gT12_re * b2_re;
    	B1_re -= gT12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += gT10_re * b0_im;
    	B1_im += gT10_im * b0_re;
    	B1_im += gT11_re * b1_im;
    	B1_im += gT11_im * b1_re;
    	B1_im += gT12_re * b2_im;
    	B1_im += gT12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += gT20_re * a0_re;
    	A2_re -= gT20_im * a0_im;
    	A2_re += gT21_re * a1_re;
    	A2_re -= gT21_im * a1_im;
    	A2_re += gT22_re * a2_re;
    	A2_re -= gT22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += gT20_re * a0_im;
    	A2_im += gT20_im * a0_re;
    	A2_im += gT21_re * a1_im;
    	A2_im += gT21_im * a1_re;
    	A2_im += gT22_re * a2_im;
    	A2_im += gT22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += gT20_re * b0_re;
    	B2_re -= gT20_im * b0_im;
    	B2_re += gT21_re * b1_re;
    	B2_re -= gT21_im * b1_im;
    	B2_re += gT22_re * b2_re;
    	B2_re -= gT22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += gT20_re * b0_im;
    	B2_im += gT20_im * b0_re;
    	B2_im += gT21_re * b1_im;
    	B2_im += gT21_im * b1_re;
    	B2_im += gT22_re * b2_im;
    	B2_im += gT22_im * b2_re;
    	
    
    	o2_00_re += A0_re;
    	o2_00_im += A0_im;
    	o2_10_re += B0_re;
    	o2_10_im += B0_im;
    	o2_20_re -= B0_re;
    	o2_20_im -= B0_im;
    	o2_30_re += A0_re;
    	o2_30_im += A0_im;
    	
    	o2_01_re += A1_re;
    	o2_01_im += A1_im;
    	o2_11_re += B1_re;
    	o2_11_im += B1_im;
    	o2_21_re -= B1_re;
    	o2_21_im -= B1_im;
    	o2_31_re += A1_re;
    	o2_31_im += A1_im;
    	
    	o2_02_re += A2_re;
    	o2_02_im += A2_im;
    	o2_12_re += B2_re;
    	o2_12_im += B2_im;
    	o2_22_re -= B2_re;
    	o2_22_im -= B2_im;
    	o2_32_re += A2_re;
    	o2_32_im += A2_im;
    	
    }
}

{
    // Projector P2-
    // 1 0 -i 0 
    // 0 1 0 i 
    // i 0 1 0 
    // 0 -i 0 1 
    
    int sp_idx = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
    int ga_idx = sid;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE0TEX, 4, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(4);
    
    {// read the first flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += g00_re * a0_re;
    	A0_re -= g00_im * a0_im;
    	A0_re += g01_re * a1_re;
    	A0_re -= g01_im * a1_im;
    	A0_re += g02_re * a2_re;
    	A0_re -= g02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += g00_re * a0_im;
    	A0_im += g00_im * a0_re;
    	A0_im += g01_re * a1_im;
    	A0_im += g01_im * a1_re;
    	A0_im += g02_re * a2_im;
    	A0_im += g02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += g00_re * b0_re;
    	B0_re -= g00_im * b0_im;
    	B0_re += g01_re * b1_re;
    	B0_re -= g01_im * b1_im;
    	B0_re += g02_re * b2_re;
    	B0_re -= g02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += g00_re * b0_im;
    	B0_im += g00_im * b0_re;
    	B0_im += g01_re * b1_im;
    	B0_im += g01_im * b1_re;
    	B0_im += g02_re * b2_im;
    	B0_im += g02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += g10_re * a0_re;
    	A1_re -= g10_im * a0_im;
    	A1_re += g11_re * a1_re;
    	A1_re -= g11_im * a1_im;
    	A1_re += g12_re * a2_re;
    	A1_re -= g12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += g10_re * a0_im;
    	A1_im += g10_im * a0_re;
    	A1_im += g11_re * a1_im;
    	A1_im += g11_im * a1_re;
    	A1_im += g12_re * a2_im;
    	A1_im += g12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += g10_re * b0_re;
    	B1_re -= g10_im * b0_im;
    	B1_re += g11_re * b1_re;
    	B1_re -= g11_im * b1_im;
    	B1_re += g12_re * b2_re;
    	B1_re -= g12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += g10_re * b0_im;
    	B1_im += g10_im * b0_re;
    	B1_im += g11_re * b1_im;
    	B1_im += g11_im * b1_re;
    	B1_im += g12_re * b2_im;
    	B1_im += g12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += g20_re * a0_re;
    	A2_re -= g20_im * a0_im;
    	A2_re += g21_re * a1_re;
    	A2_re -= g21_im * a1_im;
    	A2_re += g22_re * a2_re;
    	A2_re -= g22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += g20_re * a0_im;
    	A2_im += g20_im * a0_re;
    	A2_im += g21_re * a1_im;
    	A2_im += g21_im * a1_re;
    	A2_im += g22_re * a2_im;
    	A2_im += g22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += g20_re * b0_re;
    	B2_re -= g20_im * b0_im;
    	B2_re += g21_re * b1_re;
    	B2_re -= g21_im * b1_im;
    	B2_re += g22_re * b2_re;
    	B2_re -= g22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += g20_re * b0_im;
    	B2_im += g20_im * b0_re;
    	B2_im += g21_re * b1_im;
    	B2_im += g21_im * b1_re;
    	B2_im += g22_re * b2_im;
    	B2_im += g22_im * b2_re;
    	
    
    	o1_00_re += A0_re;
    	o1_00_im += A0_im;
    	o1_10_re += B0_re;
    	o1_10_im += B0_im;
    	o1_20_re -= A0_im;
    	o1_20_im += A0_re;
    	o1_30_re += B0_im;
    	o1_30_im -= B0_re;
    	
    	o1_01_re += A1_re;
    	o1_01_im += A1_im;
    	o1_11_re += B1_re;
    	o1_11_im += B1_im;
    	o1_21_re -= A1_im;
    	o1_21_im += A1_re;
    	o1_31_re += B1_im;
    	o1_31_im -= B1_re;
    	
    	o1_02_re += A2_re;
    	o1_02_im += A2_im;
    	o1_12_re += B2_re;
    	o1_12_im += B2_im;
    	o1_22_re -= A2_im;
    	o1_22_im += A2_re;
    	o1_32_re += B2_im;
    	o1_32_im -= B2_re;
    	
    }
    {// read the second flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += g00_re * a0_re;
    	A0_re -= g00_im * a0_im;
    	A0_re += g01_re * a1_re;
    	A0_re -= g01_im * a1_im;
    	A0_re += g02_re * a2_re;
    	A0_re -= g02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += g00_re * a0_im;
    	A0_im += g00_im * a0_re;
    	A0_im += g01_re * a1_im;
    	A0_im += g01_im * a1_re;
    	A0_im += g02_re * a2_im;
    	A0_im += g02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += g00_re * b0_re;
    	B0_re -= g00_im * b0_im;
    	B0_re += g01_re * b1_re;
    	B0_re -= g01_im * b1_im;
    	B0_re += g02_re * b2_re;
    	B0_re -= g02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += g00_re * b0_im;
    	B0_im += g00_im * b0_re;
    	B0_im += g01_re * b1_im;
    	B0_im += g01_im * b1_re;
    	B0_im += g02_re * b2_im;
    	B0_im += g02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += g10_re * a0_re;
    	A1_re -= g10_im * a0_im;
    	A1_re += g11_re * a1_re;
    	A1_re -= g11_im * a1_im;
    	A1_re += g12_re * a2_re;
    	A1_re -= g12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += g10_re * a0_im;
    	A1_im += g10_im * a0_re;
    	A1_im += g11_re * a1_im;
    	A1_im += g11_im * a1_re;
    	A1_im += g12_re * a2_im;
    	A1_im += g12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += g10_re * b0_re;
    	B1_re -= g10_im * b0_im;
    	B1_re += g11_re * b1_re;
    	B1_re -= g11_im * b1_im;
    	B1_re += g12_re * b2_re;
    	B1_re -= g12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += g10_re * b0_im;
    	B1_im += g10_im * b0_re;
    	B1_im += g11_re * b1_im;
    	B1_im += g11_im * b1_re;
    	B1_im += g12_re * b2_im;
    	B1_im += g12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += g20_re * a0_re;
    	A2_re -= g20_im * a0_im;
    	A2_re += g21_re * a1_re;
    	A2_re -= g21_im * a1_im;
    	A2_re += g22_re * a2_re;
    	A2_re -= g22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += g20_re * a0_im;
    	A2_im += g20_im * a0_re;
    	A2_im += g21_re * a1_im;
    	A2_im += g21_im * a1_re;
    	A2_im += g22_re * a2_im;
    	A2_im += g22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += g20_re * b0_re;
    	B2_re -= g20_im * b0_im;
    	B2_re += g21_re * b1_re;
    	B2_re -= g21_im * b1_im;
    	B2_re += g22_re * b2_re;
    	B2_re -= g22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += g20_re * b0_im;
    	B2_im += g20_im * b0_re;
    	B2_im += g21_re * b1_im;
    	B2_im += g21_im * b1_re;
    	B2_im += g22_re * b2_im;
    	B2_im += g22_im * b2_re;
    	
    
    	o2_00_re += A0_re;
    	o2_00_im += A0_im;
    	o2_10_re += B0_re;
    	o2_10_im += B0_im;
    	o2_20_re -= A0_im;
    	o2_20_im += A0_re;
    	o2_30_re += B0_im;
    	o2_30_im -= B0_re;
    	
    	o2_01_re += A1_re;
    	o2_01_im += A1_im;
    	o2_11_re += B1_re;
    	o2_11_im += B1_im;
    	o2_21_re -= A1_im;
    	o2_21_im += A1_re;
    	o2_31_re += B1_im;
    	o2_31_im -= B1_re;
    	
    	o2_02_re += A2_re;
    	o2_02_im += A2_im;
    	o2_12_re += B2_re;
    	o2_12_im += B2_im;
    	o2_22_re -= A2_im;
    	o2_22_im += A2_re;
    	o2_32_re += B2_im;
    	o2_32_im -= B2_re;
    	
    }
}

{
    // Projector P2+
    // 1 0 i 0 
    // 0 1 0 -i 
    // -i 0 1 0 
    // 0 i 0 1 
    
    int sp_idx = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
    int ga_idx = sp_idx;
    
    // read gauge matrix from device memory
    READ_GAUGE_MATRIX(G, GAUGE1TEX, 5, ga_idx, ga_stride);
    
    // reconstruct gauge matrix
    RECONSTRUCT_GAUGE_MATRIX(5);
    
    {// read the first flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx, sp_idx);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += gT00_re * a0_re;
    	A0_re -= gT00_im * a0_im;
    	A0_re += gT01_re * a1_re;
    	A0_re -= gT01_im * a1_im;
    	A0_re += gT02_re * a2_re;
    	A0_re -= gT02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += gT00_re * a0_im;
    	A0_im += gT00_im * a0_re;
    	A0_im += gT01_re * a1_im;
    	A0_im += gT01_im * a1_re;
    	A0_im += gT02_re * a2_im;
    	A0_im += gT02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += gT00_re * b0_re;
    	B0_re -= gT00_im * b0_im;
    	B0_re += gT01_re * b1_re;
    	B0_re -= gT01_im * b1_im;
    	B0_re += gT02_re * b2_re;
    	B0_re -= gT02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += gT00_re * b0_im;
    	B0_im += gT00_im * b0_re;
    	B0_im += gT01_re * b1_im;
    	B0_im += gT01_im * b1_re;
    	B0_im += gT02_re * b2_im;
    	B0_im += gT02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += gT10_re * a0_re;
    	A1_re -= gT10_im * a0_im;
    	A1_re += gT11_re * a1_re;
    	A1_re -= gT11_im * a1_im;
    	A1_re += gT12_re * a2_re;
    	A1_re -= gT12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += gT10_re * a0_im;
    	A1_im += gT10_im * a0_re;
    	A1_im += gT11_re * a1_im;
    	A1_im += gT11_im * a1_re;
    	A1_im += gT12_re * a2_im;
    	A1_im += gT12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += gT10_re * b0_re;
    	B1_re -= gT10_im * b0_im;
    	B1_re += gT11_re * b1_re;
    	B1_re -= gT11_im * b1_im;
    	B1_re += gT12_re * b2_re;
    	B1_re -= gT12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += gT10_re * b0_im;
    	B1_im += gT10_im * b0_re;
    	B1_im += gT11_re * b1_im;
    	B1_im += gT11_im * b1_re;
    	B1_im += gT12_re * b2_im;
    	B1_im += gT12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += gT20_re * a0_re;
    	A2_re -= gT20_im * a0_im;
    	A2_re += gT21_re * a1_re;
    	A2_re -= gT21_im * a1_im;
    	A2_re += gT22_re * a2_re;
    	A2_re -= gT22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += gT20_re * a0_im;
    	A2_im += gT20_im * a0_re;
    	A2_im += gT21_re * a1_im;
    	A2_im += gT21_im * a1_re;
    	A2_im += gT22_re * a2_im;
    	A2_im += gT22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += gT20_re * b0_re;
    	B2_re -= gT20_im * b0_im;
    	B2_re += gT21_re * b1_re;
    	B2_re -= gT21_im * b1_im;
    	B2_re += gT22_re * b2_re;
    	B2_re -= gT22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += gT20_re * b0_im;
    	B2_im += gT20_im * b0_re;
    	B2_im += gT21_re * b1_im;
    	B2_im += gT21_im * b1_re;
    	B2_im += gT22_re * b2_im;
    	B2_im += gT22_im * b2_re;
    	
    
    	o1_00_re += A0_re;
    	o1_00_im += A0_im;
    	o1_10_re += B0_re;
    	o1_10_im += B0_im;
    	o1_20_re += A0_im;
    	o1_20_im -= A0_re;
    	o1_30_re -= B0_im;
    	o1_30_im += B0_re;
    	
    	o1_01_re += A1_re;
    	o1_01_im += A1_im;
    	o1_11_re += B1_re;
    	o1_11_im += B1_im;
    	o1_21_re += A1_im;
    	o1_21_im -= A1_re;
    	o1_31_re -= B1_im;
    	o1_31_im += B1_re;
    	
    	o1_02_re += A2_re;
    	o1_02_im += A2_im;
    	o1_12_re += B2_re;
    	o1_12_im += B2_im;
    	o1_22_re += A2_im;
    	o1_22_im -= A2_re;
    	o1_32_re -= B2_im;
    	o1_32_im += B2_re;
    	
    }
    {// read the second flavor spinor from device memory
    	READ_SPINOR(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
    
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
    	spinorFloat A0_re = 0;
    	A0_re += gT00_re * a0_re;
    	A0_re -= gT00_im * a0_im;
    	A0_re += gT01_re * a1_re;
    	A0_re -= gT01_im * a1_im;
    	A0_re += gT02_re * a2_re;
    	A0_re -= gT02_im * a2_im;
    	spinorFloat A0_im = 0;
    	A0_im += gT00_re * a0_im;
    	A0_im += gT00_im * a0_re;
    	A0_im += gT01_re * a1_im;
    	A0_im += gT01_im * a1_re;
    	A0_im += gT02_re * a2_im;
    	A0_im += gT02_im * a2_re;
    	spinorFloat B0_re = 0;
    	B0_re += gT00_re * b0_re;
    	B0_re -= gT00_im * b0_im;
    	B0_re += gT01_re * b1_re;
    	B0_re -= gT01_im * b1_im;
    	B0_re += gT02_re * b2_re;
    	B0_re -= gT02_im * b2_im;
    	spinorFloat B0_im = 0;
    	B0_im += gT00_re * b0_im;
    	B0_im += gT00_im * b0_re;
    	B0_im += gT01_re * b1_im;
    	B0_im += gT01_im * b1_re;
    	B0_im += gT02_re * b2_im;
    	B0_im += gT02_im * b2_re;
    	
    	// multiply row 1
    	spinorFloat A1_re = 0;
    	A1_re += gT10_re * a0_re;
    	A1_re -= gT10_im * a0_im;
    	A1_re += gT11_re * a1_re;
    	A1_re -= gT11_im * a1_im;
    	A1_re += gT12_re * a2_re;
    	A1_re -= gT12_im * a2_im;
    	spinorFloat A1_im = 0;
    	A1_im += gT10_re * a0_im;
    	A1_im += gT10_im * a0_re;
    	A1_im += gT11_re * a1_im;
    	A1_im += gT11_im * a1_re;
    	A1_im += gT12_re * a2_im;
    	A1_im += gT12_im * a2_re;
    	spinorFloat B1_re = 0;
    	B1_re += gT10_re * b0_re;
    	B1_re -= gT10_im * b0_im;
    	B1_re += gT11_re * b1_re;
    	B1_re -= gT11_im * b1_im;
    	B1_re += gT12_re * b2_re;
    	B1_re -= gT12_im * b2_im;
    	spinorFloat B1_im = 0;
    	B1_im += gT10_re * b0_im;
    	B1_im += gT10_im * b0_re;
    	B1_im += gT11_re * b1_im;
    	B1_im += gT11_im * b1_re;
    	B1_im += gT12_re * b2_im;
    	B1_im += gT12_im * b2_re;
    	
    	// multiply row 2
    	spinorFloat A2_re = 0;
    	A2_re += gT20_re * a0_re;
    	A2_re -= gT20_im * a0_im;
    	A2_re += gT21_re * a1_re;
    	A2_re -= gT21_im * a1_im;
    	A2_re += gT22_re * a2_re;
    	A2_re -= gT22_im * a2_im;
    	spinorFloat A2_im = 0;
    	A2_im += gT20_re * a0_im;
    	A2_im += gT20_im * a0_re;
    	A2_im += gT21_re * a1_im;
    	A2_im += gT21_im * a1_re;
    	A2_im += gT22_re * a2_im;
    	A2_im += gT22_im * a2_re;
    	spinorFloat B2_re = 0;
    	B2_re += gT20_re * b0_re;
    	B2_re -= gT20_im * b0_im;
    	B2_re += gT21_re * b1_re;
    	B2_re -= gT21_im * b1_im;
    	B2_re += gT22_re * b2_re;
    	B2_re -= gT22_im * b2_im;
    	spinorFloat B2_im = 0;
    	B2_im += gT20_re * b0_im;
    	B2_im += gT20_im * b0_re;
    	B2_im += gT21_re * b1_im;
    	B2_im += gT21_im * b1_re;
    	B2_im += gT22_re * b2_im;
    	B2_im += gT22_im * b2_re;
    	
    
    	o2_00_re += A0_re;
    	o2_00_im += A0_im;
    	o2_10_re += B0_re;
    	o2_10_im += B0_im;
    	o2_20_re += A0_im;
    	o2_20_im -= A0_re;
    	o2_30_re -= B0_im;
    	o2_30_im += B0_re;
    	
    	o2_01_re += A1_re;
    	o2_01_im += A1_im;
    	o2_11_re += B1_re;
    	o2_11_im += B1_im;
    	o2_21_re += A1_im;
    	o2_21_im -= A1_re;
    	o2_31_re -= B1_im;
    	o2_31_im += B1_re;
    	
    	o2_02_re += A2_re;
    	o2_02_im += A2_im;
    	o2_12_re += B2_re;
    	o2_12_im += B2_im;
    	o2_22_re += A2_im;
    	o2_22_im -= A2_re;
    	o2_32_re -= B2_im;
    	o2_32_im += B2_re;
    	
    }
}

{
    // Projector P3-
    // 0 0 0 0 
    // 0 0 0 0 
    // 0 0 2 0 
    // 0 0 0 2 
    
    int sp_idx = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
    int ga_idx = sid;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        {// read the first flavor spinor from device memory
        	READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
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
        	spinorFloat A0_re = a0_re; 	spinorFloat A0_im = a0_im;
        	spinorFloat B0_re = b0_re; 	spinorFloat B0_im = b0_im;
        	spinorFloat A1_re = a1_re; 	spinorFloat A1_im = a1_im;
        	spinorFloat B1_re = b1_re; 	spinorFloat B1_im = b1_im;
        	spinorFloat A2_re = a2_re; 	spinorFloat A2_im = a2_im;
        	spinorFloat B2_re = b2_re; 	spinorFloat B2_im = b2_im;
        	
        
        	o1_20_re += A0_re;
        	o1_20_im += A0_im;
        	o1_30_re += B0_re;
        	o1_30_im += B0_im;
        	
        	o1_21_re += A1_re;
        	o1_21_im += A1_im;
        	o1_31_re += B1_re;
        	o1_31_im += B1_im;
        	
        	o1_22_re += A2_re;
        	o1_22_im += A2_im;
        	o1_32_re += B2_re;
        	o1_32_im += B2_im;
        	
        }
        {// read the second flavor spinor from device memory
        	READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
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
        	spinorFloat A0_re = a0_re; 	spinorFloat A0_im = a0_im;
        	spinorFloat B0_re = b0_re; 	spinorFloat B0_im = b0_im;
        	spinorFloat A1_re = a1_re; 	spinorFloat A1_im = a1_im;
        	spinorFloat B1_re = b1_re; 	spinorFloat B1_im = b1_im;
        	spinorFloat A2_re = a2_re; 	spinorFloat A2_im = a2_im;
        	spinorFloat B2_re = b2_re; 	spinorFloat B2_im = b2_im;
        	
        
        	o2_20_re += A0_re;
        	o2_20_im += A0_im;
        	o2_30_re += B0_re;
        	o2_30_im += B0_im;
        	
        	o2_21_re += A1_re;
        	o2_21_im += A1_im;
        	o2_31_re += B1_re;
        	o2_31_im += B1_im;
        	
        	o2_22_re += A2_re;
        	o2_22_im += A2_im;
        	o2_32_re += B2_re;
        	o2_32_im += B2_im;
        	
        }
    }
    else {
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(G, GAUGE0TEX, 6, ga_idx, ga_stride);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(6);
        
        {// read the first flavor spinor from device memory
        	READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx, sp_idx);
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
        	spinorFloat A0_re = 0;
        	A0_re += g00_re * a0_re;
        	A0_re -= g00_im * a0_im;
        	A0_re += g01_re * a1_re;
        	A0_re -= g01_im * a1_im;
        	A0_re += g02_re * a2_re;
        	A0_re -= g02_im * a2_im;
        	spinorFloat A0_im = 0;
        	A0_im += g00_re * a0_im;
        	A0_im += g00_im * a0_re;
        	A0_im += g01_re * a1_im;
        	A0_im += g01_im * a1_re;
        	A0_im += g02_re * a2_im;
        	A0_im += g02_im * a2_re;
        	spinorFloat B0_re = 0;
        	B0_re += g00_re * b0_re;
        	B0_re -= g00_im * b0_im;
        	B0_re += g01_re * b1_re;
        	B0_re -= g01_im * b1_im;
        	B0_re += g02_re * b2_re;
        	B0_re -= g02_im * b2_im;
        	spinorFloat B0_im = 0;
        	B0_im += g00_re * b0_im;
        	B0_im += g00_im * b0_re;
        	B0_im += g01_re * b1_im;
        	B0_im += g01_im * b1_re;
        	B0_im += g02_re * b2_im;
        	B0_im += g02_im * b2_re;
        	
        	// multiply row 1
        	spinorFloat A1_re = 0;
        	A1_re += g10_re * a0_re;
        	A1_re -= g10_im * a0_im;
        	A1_re += g11_re * a1_re;
        	A1_re -= g11_im * a1_im;
        	A1_re += g12_re * a2_re;
        	A1_re -= g12_im * a2_im;
        	spinorFloat A1_im = 0;
        	A1_im += g10_re * a0_im;
        	A1_im += g10_im * a0_re;
        	A1_im += g11_re * a1_im;
        	A1_im += g11_im * a1_re;
        	A1_im += g12_re * a2_im;
        	A1_im += g12_im * a2_re;
        	spinorFloat B1_re = 0;
        	B1_re += g10_re * b0_re;
        	B1_re -= g10_im * b0_im;
        	B1_re += g11_re * b1_re;
        	B1_re -= g11_im * b1_im;
        	B1_re += g12_re * b2_re;
        	B1_re -= g12_im * b2_im;
        	spinorFloat B1_im = 0;
        	B1_im += g10_re * b0_im;
        	B1_im += g10_im * b0_re;
        	B1_im += g11_re * b1_im;
        	B1_im += g11_im * b1_re;
        	B1_im += g12_re * b2_im;
        	B1_im += g12_im * b2_re;
        	
        	// multiply row 2
        	spinorFloat A2_re = 0;
        	A2_re += g20_re * a0_re;
        	A2_re -= g20_im * a0_im;
        	A2_re += g21_re * a1_re;
        	A2_re -= g21_im * a1_im;
        	A2_re += g22_re * a2_re;
        	A2_re -= g22_im * a2_im;
        	spinorFloat A2_im = 0;
        	A2_im += g20_re * a0_im;
        	A2_im += g20_im * a0_re;
        	A2_im += g21_re * a1_im;
        	A2_im += g21_im * a1_re;
        	A2_im += g22_re * a2_im;
        	A2_im += g22_im * a2_re;
        	spinorFloat B2_re = 0;
        	B2_re += g20_re * b0_re;
        	B2_re -= g20_im * b0_im;
        	B2_re += g21_re * b1_re;
        	B2_re -= g21_im * b1_im;
        	B2_re += g22_re * b2_re;
        	B2_re -= g22_im * b2_im;
        	spinorFloat B2_im = 0;
        	B2_im += g20_re * b0_im;
        	B2_im += g20_im * b0_re;
        	B2_im += g21_re * b1_im;
        	B2_im += g21_im * b1_re;
        	B2_im += g22_re * b2_im;
        	B2_im += g22_im * b2_re;
        	
        
        	o1_20_re += A0_re;
        	o1_20_im += A0_im;
        	o1_30_re += B0_re;
        	o1_30_im += B0_im;
        	
        	o1_21_re += A1_re;
        	o1_21_im += A1_im;
        	o1_31_re += B1_re;
        	o1_31_im += B1_im;
        	
        	o1_22_re += A2_re;
        	o1_22_im += A2_im;
        	o1_32_re += B2_re;
        	o1_32_im += B2_im;
        	
        }
        {// read the second flavor spinor from device memory
        	READ_SPINOR_DOWN(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
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
        	spinorFloat A0_re = 0;
        	A0_re += g00_re * a0_re;
        	A0_re -= g00_im * a0_im;
        	A0_re += g01_re * a1_re;
        	A0_re -= g01_im * a1_im;
        	A0_re += g02_re * a2_re;
        	A0_re -= g02_im * a2_im;
        	spinorFloat A0_im = 0;
        	A0_im += g00_re * a0_im;
        	A0_im += g00_im * a0_re;
        	A0_im += g01_re * a1_im;
        	A0_im += g01_im * a1_re;
        	A0_im += g02_re * a2_im;
        	A0_im += g02_im * a2_re;
        	spinorFloat B0_re = 0;
        	B0_re += g00_re * b0_re;
        	B0_re -= g00_im * b0_im;
        	B0_re += g01_re * b1_re;
        	B0_re -= g01_im * b1_im;
        	B0_re += g02_re * b2_re;
        	B0_re -= g02_im * b2_im;
        	spinorFloat B0_im = 0;
        	B0_im += g00_re * b0_im;
        	B0_im += g00_im * b0_re;
        	B0_im += g01_re * b1_im;
        	B0_im += g01_im * b1_re;
        	B0_im += g02_re * b2_im;
        	B0_im += g02_im * b2_re;
        	
        	// multiply row 1
        	spinorFloat A1_re = 0;
        	A1_re += g10_re * a0_re;
        	A1_re -= g10_im * a0_im;
        	A1_re += g11_re * a1_re;
        	A1_re -= g11_im * a1_im;
        	A1_re += g12_re * a2_re;
        	A1_re -= g12_im * a2_im;
        	spinorFloat A1_im = 0;
        	A1_im += g10_re * a0_im;
        	A1_im += g10_im * a0_re;
        	A1_im += g11_re * a1_im;
        	A1_im += g11_im * a1_re;
        	A1_im += g12_re * a2_im;
        	A1_im += g12_im * a2_re;
        	spinorFloat B1_re = 0;
        	B1_re += g10_re * b0_re;
        	B1_re -= g10_im * b0_im;
        	B1_re += g11_re * b1_re;
        	B1_re -= g11_im * b1_im;
        	B1_re += g12_re * b2_re;
        	B1_re -= g12_im * b2_im;
        	spinorFloat B1_im = 0;
        	B1_im += g10_re * b0_im;
        	B1_im += g10_im * b0_re;
        	B1_im += g11_re * b1_im;
        	B1_im += g11_im * b1_re;
        	B1_im += g12_re * b2_im;
        	B1_im += g12_im * b2_re;
        	
        	// multiply row 2
        	spinorFloat A2_re = 0;
        	A2_re += g20_re * a0_re;
        	A2_re -= g20_im * a0_im;
        	A2_re += g21_re * a1_re;
        	A2_re -= g21_im * a1_im;
        	A2_re += g22_re * a2_re;
        	A2_re -= g22_im * a2_im;
        	spinorFloat A2_im = 0;
        	A2_im += g20_re * a0_im;
        	A2_im += g20_im * a0_re;
        	A2_im += g21_re * a1_im;
        	A2_im += g21_im * a1_re;
        	A2_im += g22_re * a2_im;
        	A2_im += g22_im * a2_re;
        	spinorFloat B2_re = 0;
        	B2_re += g20_re * b0_re;
        	B2_re -= g20_im * b0_im;
        	B2_re += g21_re * b1_re;
        	B2_re -= g21_im * b1_im;
        	B2_re += g22_re * b2_re;
        	B2_re -= g22_im * b2_im;
        	spinorFloat B2_im = 0;
        	B2_im += g20_re * b0_im;
        	B2_im += g20_im * b0_re;
        	B2_im += g21_re * b1_im;
        	B2_im += g21_im * b1_re;
        	B2_im += g22_re * b2_im;
        	B2_im += g22_im * b2_re;
        	
        
        	o2_20_re += A0_re;
        	o2_20_im += A0_im;
        	o2_30_re += B0_re;
        	o2_30_im += B0_im;
        	
        	o2_21_re += A1_re;
        	o2_21_im += A1_im;
        	o2_31_re += B1_re;
        	o2_31_im += B1_im;
        	
        	o2_22_re += A2_re;
        	o2_22_im += A2_im;
        	o2_32_re += B2_re;
        	o2_32_im += B2_im;
        	
        }
    }
}

{
    // Projector P3+
    // 2 0 0 0 
    // 0 2 0 0 
    // 0 0 0 0 
    // 0 0 0 0 
    
    int sp_idx = ((x4==0)    ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
    int ga_idx = sp_idx;
    
    if (gauge_fixed && ga_idx < X4X3X2X1hmX3X2X1h) {
        {// read the first flavor spinor from device memory
        	READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
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
        	spinorFloat A0_re = a0_re; 	spinorFloat A0_im = a0_im;
        	spinorFloat B0_re = b0_re; 	spinorFloat B0_im = b0_im;
        	spinorFloat A1_re = a1_re; 	spinorFloat A1_im = a1_im;
        	spinorFloat B1_re = b1_re; 	spinorFloat B1_im = b1_im;
        	spinorFloat A2_re = a2_re; 	spinorFloat A2_im = a2_im;
        	spinorFloat B2_re = b2_re; 	spinorFloat B2_im = b2_im;
        	
        
        	o1_00_re += A0_re;
        	o1_00_im += A0_im;
        	o1_10_re += B0_re;
        	o1_10_im += B0_im;
        	
        	o1_01_re += A1_re;
        	o1_01_im += A1_im;
        	o1_11_re += B1_re;
        	o1_11_im += B1_im;
        	
        	o1_02_re += A2_re;
        	o1_02_im += A2_im;
        	o1_12_re += B2_re;
        	o1_12_im += B2_im;
        	
        }
        {// read the second flavor spinor from device memory
        	READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
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
        	spinorFloat A0_re = a0_re; 	spinorFloat A0_im = a0_im;
        	spinorFloat B0_re = b0_re; 	spinorFloat B0_im = b0_im;
        	spinorFloat A1_re = a1_re; 	spinorFloat A1_im = a1_im;
        	spinorFloat B1_re = b1_re; 	spinorFloat B1_im = b1_im;
        	spinorFloat A2_re = a2_re; 	spinorFloat A2_im = a2_im;
        	spinorFloat B2_re = b2_re; 	spinorFloat B2_im = b2_im;
        	
        
        	o2_00_re += A0_re;
        	o2_00_im += A0_im;
        	o2_10_re += B0_re;
        	o2_10_im += B0_im;
        	
        	o2_01_re += A1_re;
        	o2_01_im += A1_im;
        	o2_11_re += B1_re;
        	o2_11_im += B1_im;
        	
        	o2_02_re += A2_re;
        	o2_02_im += A2_im;
        	o2_12_re += B2_re;
        	o2_12_im += B2_im;
        	
        }
    }
    else {
        // read gauge matrix from device memory
        READ_GAUGE_MATRIX(G, GAUGE1TEX, 7, ga_idx, ga_stride);
        
        // reconstruct gauge matrix
        RECONSTRUCT_GAUGE_MATRIX(7);
        
        {// read the first flavor spinor from device memory
        	READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx, sp_idx);
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
        	spinorFloat A0_re = 0;
        	A0_re += gT00_re * a0_re;
        	A0_re -= gT00_im * a0_im;
        	A0_re += gT01_re * a1_re;
        	A0_re -= gT01_im * a1_im;
        	A0_re += gT02_re * a2_re;
        	A0_re -= gT02_im * a2_im;
        	spinorFloat A0_im = 0;
        	A0_im += gT00_re * a0_im;
        	A0_im += gT00_im * a0_re;
        	A0_im += gT01_re * a1_im;
        	A0_im += gT01_im * a1_re;
        	A0_im += gT02_re * a2_im;
        	A0_im += gT02_im * a2_re;
        	spinorFloat B0_re = 0;
        	B0_re += gT00_re * b0_re;
        	B0_re -= gT00_im * b0_im;
        	B0_re += gT01_re * b1_re;
        	B0_re -= gT01_im * b1_im;
        	B0_re += gT02_re * b2_re;
        	B0_re -= gT02_im * b2_im;
        	spinorFloat B0_im = 0;
        	B0_im += gT00_re * b0_im;
        	B0_im += gT00_im * b0_re;
        	B0_im += gT01_re * b1_im;
        	B0_im += gT01_im * b1_re;
        	B0_im += gT02_re * b2_im;
        	B0_im += gT02_im * b2_re;
        	
        	// multiply row 1
        	spinorFloat A1_re = 0;
        	A1_re += gT10_re * a0_re;
        	A1_re -= gT10_im * a0_im;
        	A1_re += gT11_re * a1_re;
        	A1_re -= gT11_im * a1_im;
        	A1_re += gT12_re * a2_re;
        	A1_re -= gT12_im * a2_im;
        	spinorFloat A1_im = 0;
        	A1_im += gT10_re * a0_im;
        	A1_im += gT10_im * a0_re;
        	A1_im += gT11_re * a1_im;
        	A1_im += gT11_im * a1_re;
        	A1_im += gT12_re * a2_im;
        	A1_im += gT12_im * a2_re;
        	spinorFloat B1_re = 0;
        	B1_re += gT10_re * b0_re;
        	B1_re -= gT10_im * b0_im;
        	B1_re += gT11_re * b1_re;
        	B1_re -= gT11_im * b1_im;
        	B1_re += gT12_re * b2_re;
        	B1_re -= gT12_im * b2_im;
        	spinorFloat B1_im = 0;
        	B1_im += gT10_re * b0_im;
        	B1_im += gT10_im * b0_re;
        	B1_im += gT11_re * b1_im;
        	B1_im += gT11_im * b1_re;
        	B1_im += gT12_re * b2_im;
        	B1_im += gT12_im * b2_re;
        	
        	// multiply row 2
        	spinorFloat A2_re = 0;
        	A2_re += gT20_re * a0_re;
        	A2_re -= gT20_im * a0_im;
        	A2_re += gT21_re * a1_re;
        	A2_re -= gT21_im * a1_im;
        	A2_re += gT22_re * a2_re;
        	A2_re -= gT22_im * a2_im;
        	spinorFloat A2_im = 0;
        	A2_im += gT20_re * a0_im;
        	A2_im += gT20_im * a0_re;
        	A2_im += gT21_re * a1_im;
        	A2_im += gT21_im * a1_re;
        	A2_im += gT22_re * a2_im;
        	A2_im += gT22_im * a2_re;
        	spinorFloat B2_re = 0;
        	B2_re += gT20_re * b0_re;
        	B2_re -= gT20_im * b0_im;
        	B2_re += gT21_re * b1_re;
        	B2_re -= gT21_im * b1_im;
        	B2_re += gT22_re * b2_re;
        	B2_re -= gT22_im * b2_im;
        	spinorFloat B2_im = 0;
        	B2_im += gT20_re * b0_im;
        	B2_im += gT20_im * b0_re;
        	B2_im += gT21_re * b1_im;
        	B2_im += gT21_im * b1_re;
        	B2_im += gT22_re * b2_im;
        	B2_im += gT22_im * b2_re;
        	
        
        	o1_00_re += A0_re;
        	o1_00_im += A0_im;
        	o1_10_re += B0_re;
        	o1_10_im += B0_im;
        	
        	o1_01_re += A1_re;
        	o1_01_im += A1_im;
        	o1_11_re += B1_re;
        	o1_11_im += B1_im;
        	
        	o1_02_re += A2_re;
        	o1_02_im += A2_im;
        	o1_12_re += B2_re;
        	o1_12_im += B2_im;
        	
        }
        {// read the second flavor spinor from device memory
        	READ_SPINOR_UP(SPINORTEX, sp_stride, sp_idx+fl_stride, sp_idx+fl_stride);
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
        	spinorFloat A0_re = 0;
        	A0_re += gT00_re * a0_re;
        	A0_re -= gT00_im * a0_im;
        	A0_re += gT01_re * a1_re;
        	A0_re -= gT01_im * a1_im;
        	A0_re += gT02_re * a2_re;
        	A0_re -= gT02_im * a2_im;
        	spinorFloat A0_im = 0;
        	A0_im += gT00_re * a0_im;
        	A0_im += gT00_im * a0_re;
        	A0_im += gT01_re * a1_im;
        	A0_im += gT01_im * a1_re;
        	A0_im += gT02_re * a2_im;
        	A0_im += gT02_im * a2_re;
        	spinorFloat B0_re = 0;
        	B0_re += gT00_re * b0_re;
        	B0_re -= gT00_im * b0_im;
        	B0_re += gT01_re * b1_re;
        	B0_re -= gT01_im * b1_im;
        	B0_re += gT02_re * b2_re;
        	B0_re -= gT02_im * b2_im;
        	spinorFloat B0_im = 0;
        	B0_im += gT00_re * b0_im;
        	B0_im += gT00_im * b0_re;
        	B0_im += gT01_re * b1_im;
        	B0_im += gT01_im * b1_re;
        	B0_im += gT02_re * b2_im;
        	B0_im += gT02_im * b2_re;
        	
        	// multiply row 1
        	spinorFloat A1_re = 0;
        	A1_re += gT10_re * a0_re;
        	A1_re -= gT10_im * a0_im;
        	A1_re += gT11_re * a1_re;
        	A1_re -= gT11_im * a1_im;
        	A1_re += gT12_re * a2_re;
        	A1_re -= gT12_im * a2_im;
        	spinorFloat A1_im = 0;
        	A1_im += gT10_re * a0_im;
        	A1_im += gT10_im * a0_re;
        	A1_im += gT11_re * a1_im;
        	A1_im += gT11_im * a1_re;
        	A1_im += gT12_re * a2_im;
        	A1_im += gT12_im * a2_re;
        	spinorFloat B1_re = 0;
        	B1_re += gT10_re * b0_re;
        	B1_re -= gT10_im * b0_im;
        	B1_re += gT11_re * b1_re;
        	B1_re -= gT11_im * b1_im;
        	B1_re += gT12_re * b2_re;
        	B1_re -= gT12_im * b2_im;
        	spinorFloat B1_im = 0;
        	B1_im += gT10_re * b0_im;
        	B1_im += gT10_im * b0_re;
        	B1_im += gT11_re * b1_im;
        	B1_im += gT11_im * b1_re;
        	B1_im += gT12_re * b2_im;
        	B1_im += gT12_im * b2_re;
        	
        	// multiply row 2
        	spinorFloat A2_re = 0;
        	A2_re += gT20_re * a0_re;
        	A2_re -= gT20_im * a0_im;
        	A2_re += gT21_re * a1_re;
        	A2_re -= gT21_im * a1_im;
        	A2_re += gT22_re * a2_re;
        	A2_re -= gT22_im * a2_im;
        	spinorFloat A2_im = 0;
        	A2_im += gT20_re * a0_im;
        	A2_im += gT20_im * a0_re;
        	A2_im += gT21_re * a1_im;
        	A2_im += gT21_im * a1_re;
        	A2_im += gT22_re * a2_im;
        	A2_im += gT22_im * a2_re;
        	spinorFloat B2_re = 0;
        	B2_re += gT20_re * b0_re;
        	B2_re -= gT20_im * b0_im;
        	B2_re += gT21_re * b1_re;
        	B2_re -= gT21_im * b1_im;
        	B2_re += gT22_re * b2_re;
        	B2_re -= gT22_im * b2_im;
        	spinorFloat B2_im = 0;
        	B2_im += gT20_re * b0_im;
        	B2_im += gT20_im * b0_re;
        	B2_im += gT21_re * b1_im;
        	B2_im += gT21_im * b1_re;
        	B2_im += gT22_re * b2_im;
        	B2_im += gT22_im * b2_re;
        	
        
        	o2_00_re += A0_re;
        	o2_00_im += A0_im;
        	o2_10_re += B0_re;
        	o2_10_im += B0_im;
        	
        	o2_01_re += A1_re;
        	o2_01_im += A1_im;
        	o2_11_re += B1_re;
        	o2_11_im += B1_im;
        	
        	o2_02_re += A2_re;
        	o2_02_im += A2_im;
        	o2_12_re += B2_re;
        	o2_12_im += B2_im;
        	
        }
    }
}

//Perform twist rotation first:
//(1 - i*a*gamma_5 * tau_3 + b * tau_1)
///notations : mu -> a, epsilon -> b

{
   volatile spinorFloat x1_re, x1_im, y1_re, y1_im;
   volatile spinorFloat x2_re, x2_im, y2_re, y2_im;
   
   x1_re = 0.0, x1_im = 0.0;
   y1_re = 0.0, y1_im = 0.0;
   
   x2_re = 0.0, x2_im = 0.0;
   y2_re = 0.0, y2_im = 0.0;

   //color = 0, spin = 0, 2 for each flavor:   
   // using o1:
   x1_re = o1_00_re + a * o1_20_im;
   x1_im = o1_00_im - a * o1_20_re;

   x2_re = b * o1_00_re;
   x2_im = b * o1_00_im;

   y1_re = o1_20_re + a * o1_00_im;
   y1_im = o1_20_im - a * o1_00_re;   

   y2_re = b * o1_20_re;
   y2_im = b * o1_20_im;
   // using o2:   
   x2_re += o2_00_re - a * o2_20_im;
   x2_im += o2_00_im + a * o2_20_re;

   x1_re += b * o2_00_re;
   x1_im += b * o2_00_im;

   y2_re += o2_20_re - a * o2_00_im;
   y2_im += o2_20_im + a * o2_00_re;   

   y1_re += b * o2_20_re;
   y1_im += b * o2_20_im;
   
   //store results back to output regs:
   
   o1_00_re = x1_re, o1_00_im = x1_im;   
   o1_20_re = y1_re, o1_20_im = y1_im;   

   o2_00_re = x2_re, o2_00_im = x2_im;   
   o2_20_re = y2_re, o2_20_im = y2_im;   

   //color = 0, spin = 1, 3 for each flavor:

   // using o1:
   x1_re = o1_10_re + a * o1_30_im;
   x1_im = o1_10_im - a * o1_30_re;

   x2_re = b * o1_10_re;
   x2_im = b * o1_10_im;

   y1_re = o1_30_re + a * o1_10_im;
   y1_im = o1_30_im - a * o1_10_re;   

   y2_re = b * o1_30_re;
   y2_im = b * o1_30_im;
   // using o2:   
   x2_re += o2_10_re - a * o2_30_im;
   x2_im += o2_10_im + a * o2_30_re;

   x1_re += b * o2_10_re;
   x1_im += b * o2_10_im;

   y2_re += o2_30_re - a * o2_10_im;
   y2_im += o2_30_im + a * o2_10_re;   

   y1_re += b * o2_30_re;
   y1_im += b * o2_30_im;
   
   //store results back to output regs:
   
   o1_10_re = x1_re, o1_10_im = x1_im;   
   o1_30_re = y1_re, o1_30_im = y1_im;   

   o2_10_re = x2_re, o2_10_im = x2_im;   
   o2_30_re = y2_re, o2_30_im = y2_im;   
   
   //color = 1, spin = 0, 2 for each flavor:
   
   // using o1:
   x1_re = o1_01_re + a * o1_21_im;
   x1_im = o1_01_im - a * o1_21_re;

   x2_re = b * o1_01_re;
   x2_im = b * o1_01_im;

   y1_re = o1_21_re + a * o1_01_im;
   y1_im = o1_21_im - a * o1_01_re;   

   y2_re = b * o1_21_re;
   y2_im = b * o1_21_im;
   // using o2:   
   x2_re += o2_01_re - a * o2_21_im;
   x2_im += o2_01_im + a * o2_21_re;

   x1_re += b * o2_01_re;
   x1_im += b * o2_01_im;

   y2_re += o2_21_re - a * o2_01_im;
   y2_im += o2_21_im + a * o2_01_re;   

   y1_re += b * o2_21_re;
   y1_im += b * o2_21_im;
   
   //store results back to output regs:
   
   o1_01_re = x1_re, o1_01_im = x1_im;   
   o1_21_re = y1_re, o1_21_im = y1_im;   

   o2_01_re = x2_re, o2_01_im = x2_im;   
   o2_21_re = y2_re, o2_21_im = y2_im;   
   

   //color = 1, spin = 1, 3 for each flavor:
   
   // using o1:
   x1_re = o1_11_re + a * o1_31_im;
   x1_im = o1_11_im - a * o1_31_re;

   x2_re = b * o1_11_re;
   x2_im = b * o1_11_im;

   y1_re = o1_31_re + a * o1_11_im;
   y1_im = o1_31_im - a * o1_11_re;   

   y2_re = b * o1_31_re;
   y2_im = b * o1_31_im;
   // using o2:   
   x2_re += o2_11_re - a * o2_31_im;
   x2_im += o2_11_im + a * o2_31_re;

   x1_re += b * o2_11_re;
   x1_im += b * o2_11_im;

   y2_re += o2_31_re - a * o2_11_im;
   y2_im += o2_31_im + a * o2_11_re;   

   y1_re += b * o2_31_re;
   y1_im += b * o2_31_im;
   
   //store results back to output regs:
   
   o1_11_re = x1_re, o1_11_im = x1_im;   
   o1_31_re = y1_re, o1_31_im = y1_im;   

   o2_11_re = x2_re, o2_11_im = x2_im;   
   o2_31_re = y2_re, o2_31_im = y2_im;   

   //color = 2, spin = 0, 2 for each flavor:

   // using o1:
   x1_re = o1_02_re + a * o1_22_im;
   x1_im = o1_02_im - a * o1_22_re;

   x2_re = b * o1_02_re;
   x2_im = b * o1_02_im;

   y1_re = o1_22_re + a * o1_02_im;
   y1_im = o1_22_im - a * o1_02_re;   

   y2_re = b * o1_22_re;
   y2_im = b * o1_22_im;
   // using o2:   
   x2_re += o2_02_re - a * o2_22_im;
   x2_im += o2_02_im + a * o2_22_re;

   x1_re += b * o2_02_re;
   x1_im += b * o2_02_im;

   y2_re += o2_22_re - a * o2_02_im;
   y2_im += o2_22_im + a * o2_02_re;   

   y1_re += b * o2_22_re;
   y1_im += b * o2_22_im;
   
   //store results back to output regs:
   
   o1_02_re = x1_re, o1_02_im = x1_im;   
   o1_22_re = y1_re, o1_22_im = y1_im;   

   o2_02_re = x2_re, o2_02_im = x2_im;   
   o2_22_re = y2_re, o2_22_im = y2_im;   

   //color = 2, spin = 1, 3 for each flavor:   
   
   // using o1:
   x1_re = o1_12_re + a * o1_32_im;
   x1_im = o1_12_im - a * o1_32_re;

   x2_re = b * o1_12_re;
   x2_im = b * o1_12_im;

   y1_re = o1_32_re + a * o1_12_im;
   y1_im = o1_32_im - a * o1_12_re;   

   y2_re = b * o1_32_re;
   y2_im = b * o1_32_im;
   // using o2:   
   x2_re += o2_12_re - a * o2_32_im;
   x2_im += o2_12_im + a * o2_32_re;

   x1_re += b * o2_12_re;
   x1_im += b * o2_12_im;

   y2_re += o2_32_re - a * o2_12_im;
   y2_im += o2_32_im + a * o2_12_re;   

   y1_re += b * o2_32_re;
   y1_im += b * o2_32_im;
   
   //store results back to output regs:
   
   o1_12_re = x1_re, o1_12_im = x1_im;   
   o1_32_re = y1_re, o1_32_im = y1_im;   

   o2_12_re = x2_re, o2_12_im = x2_im;   
   o2_32_re = y2_re, o2_32_im = y2_im;   
   
}

#ifndef DSLASH_XPAY
//multiply by c = 1.0 / (1.0 + a*a - b*b)
    o1_00_re *= c;
    o1_00_im *= c;
    o1_01_re *= c;
    o1_01_im *= c;
    o1_02_re *= c;
    o1_02_im *= c;

    o1_10_re *= c;
    o1_10_im *= c;
    o1_11_re *= c;
    o1_11_im *= c;
    o1_12_re *= c;
    o1_12_im *= c;

    o1_20_re *= c;
    o1_20_im *= c;
    o1_21_re *= c;
    o1_21_im *= c;
    o1_22_re *= c;
    o1_22_im *= c;

    o1_30_re *= c;
    o1_30_im *= c;
    o1_31_re *= c;
    o1_31_im *= c;
    o1_32_re *= c;
    o1_32_im *= c;


    o2_00_re *= c;
    o2_00_im *= c;
    o2_01_re *= c;
    o2_01_im *= c;
    o2_02_re *= c;
    o2_02_im *= c;

    o2_10_re *= c;
    o2_10_im *= c;
    o2_11_re *= c;
    o2_11_im *= c;
    o2_12_re *= c;
    o2_12_im *= c;

    o2_20_re *= c;
    o2_20_im *= c;
    o2_21_re *= c;
    o2_21_im *= c;
    o2_22_re *= c;
    o2_22_im *= c;

    o2_30_re *= c;
    o2_30_im *= c;
    o2_31_re *= c;
    o2_31_im *= c;
    o2_32_re *= c;
    o2_32_im *= c;

#else
//multiply by c = k / (1.0 + a*a - b*b); k = - kappa * kappa
int tmp = sid;
{
    READ_ACCUM(ACCUMTEX, sp_stride)

    #ifdef SPINOR_DOUBLE
    
    o1_00_re = c*o1_00_re + accum0.x;
    o1_00_im = c*o1_00_im + accum0.y;
    o1_01_re = c*o1_01_re + accum1.x;
    o1_01_im = c*o1_01_im + accum1.y;
    o1_02_re = c*o1_02_re + accum2.x;
    o1_02_im = c*o1_02_im + accum2.y;
    o1_10_re = c*o1_10_re + accum3.x;
    o1_10_im = c*o1_10_im + accum3.y;
    o1_11_re = c*o1_11_re + accum4.x;
    o1_11_im = c*o1_11_im + accum4.y;
    o1_12_re = c*o1_12_re + accum5.x;
    o1_12_im = c*o1_12_im + accum5.y;
    o1_20_re = c*o1_20_re + accum6.x;
    o1_20_im = c*o1_20_im + accum6.y;
    o1_21_re = c*o1_21_re + accum7.x;
    o1_21_im = c*o1_21_im + accum7.y;
    o1_22_re = c*o1_22_re + accum8.x;
    o1_22_im = c*o1_22_im + accum8.y;
    o1_30_re = c*o1_30_re + accum9.x;
    o1_30_im = c*o1_30_im + accum9.y;
    o1_31_re = c*o1_31_re + accum10.x;
    o1_31_im = c*o1_31_im + accum10.y;
    o1_32_re = c*o1_32_re + accum11.x;
    o1_32_im = c*o1_32_im + accum11.y;
    

    #else
    o1_00_re = c*o1_00_re + accum0.x;
    o1_00_im = c*o1_00_im + accum0.y;
    o1_01_re = c*o1_01_re + accum0.z;
    o1_01_im = c*o1_01_im + accum0.w;
    o1_02_re = c*o1_02_re + accum1.x;
    o1_02_im = c*o1_02_im + accum1.y;
    o1_10_re = c*o1_10_re + accum1.z;
    o1_10_im = c*o1_10_im + accum1.w;
    o1_11_re = c*o1_11_re + accum2.x;
    o1_11_im = c*o1_11_im + accum2.y;
    o1_12_re = c*o1_12_re + accum2.z;
    o1_12_im = c*o1_12_im + accum2.w;
    o1_20_re = c*o1_20_re + accum3.x;
    o1_20_im = c*o1_20_im + accum3.y;
    o1_21_re = c*o1_21_re + accum3.z;
    o1_21_im = c*o1_21_im + accum3.w;
    o1_22_re = c*o1_22_re + accum4.x;
    o1_22_im = c*o1_22_im + accum4.y;
    o1_30_re = c*o1_30_re + accum4.z;
    o1_30_im = c*o1_30_im + accum4.w;
    o1_31_re = c*o1_31_re + accum5.x;
    o1_31_im = c*o1_31_im + accum5.y;
    o1_32_re = c*o1_32_re + accum5.z;
    o1_32_im = c*o1_32_im + accum5.w;

    #endif // SPINOR_DOUBLE
}

{
    sid += fl_stride; 
    
    READ_ACCUM(ACCUMTEX, sp_stride)

    #ifdef SPINOR_DOUBLE
    o2_00_re = c*o2_00_re + accum0.x;
    o2_00_im = c*o2_00_im + accum0.y;
    o2_01_re = c*o2_01_re + accum1.x;
    o2_01_im = c*o2_01_im + accum1.y;
    o2_02_re = c*o2_02_re + accum2.x;
    o2_02_im = c*o2_02_im + accum2.y;
    o2_10_re = c*o2_10_re + accum3.x;
    o2_10_im = c*o2_10_im + accum3.y;
    o2_11_re = c*o2_11_re + accum4.x;
    o2_11_im = c*o2_11_im + accum4.y;
    o2_12_re = c*o2_12_re + accum5.x;
    o2_12_im = c*o2_12_im + accum5.y;
    o2_20_re = c*o2_20_re + accum6.x;
    o2_20_im = c*o2_20_im + accum6.y;
    o2_21_re = c*o2_21_re + accum7.x;
    o2_21_im = c*o2_21_im + accum7.y;
    o2_22_re = c*o2_22_re + accum8.x;
    o2_22_im = c*o2_22_im + accum8.y;
    o2_30_re = c*o2_30_re + accum9.x;
    o2_30_im = c*o2_30_im + accum9.y;
    o2_31_re = c*o2_31_re + accum10.x;
    o2_31_im = c*o2_31_im + accum10.y;
    o2_32_re = c*o2_32_re + accum11.x;
    o2_32_im = c*o2_32_im + accum11.y;
    #else
    o2_00_re = c*o2_00_re + accum0.x;
    o2_00_im = c*o2_00_im + accum0.y;
    o2_01_re = c*o2_01_re + accum0.z;
    o2_01_im = c*o2_01_im + accum0.w;
    o2_02_re = c*o2_02_re + accum1.x;
    o2_02_im = c*o2_02_im + accum1.y;
    o2_10_re = c*o2_10_re + accum1.z;
    o2_10_im = c*o2_10_im + accum1.w;
    o2_11_re = c*o2_11_re + accum2.x;
    o2_11_im = c*o2_11_im + accum2.y;
    o2_12_re = c*o2_12_re + accum2.z;
    o2_12_im = c*o2_12_im + accum2.w;
    o2_20_re = c*o2_20_re + accum3.x;
    o2_20_im = c*o2_20_im + accum3.y;
    o2_21_re = c*o2_21_re + accum3.z;
    o2_21_im = c*o2_21_im + accum3.w;
    o2_22_re = c*o2_22_re + accum4.x;
    o2_22_im = c*o2_22_im + accum4.y;
    o2_30_re = c*o2_30_re + accum4.z;
    o2_30_im = c*o2_30_im + accum4.w;
    o2_31_re = c*o2_31_re + accum5.x;
    o2_31_im = c*o2_31_im + accum5.y;
    o2_32_re = c*o2_32_re + accum5.z;
    o2_32_im = c*o2_32_im + accum5.w;
    #endif // SPINOR_DOUBLE
}

sid = tmp;

#endif // DSLASH_XPAY

// write spinor field back to device memory
    WRITE_FLAVOR_SPINOR();
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

