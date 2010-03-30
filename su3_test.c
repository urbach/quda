#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <util_quda.h>

#include <gauge_quda.h>
#include <spinor_quda.h>
#include <dslash_reference.h>
#include "enum_quda.h"

#define MAX_SHORT 32767
#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

inline short floatToShort(float a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

inline short doubleToShort(double a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

// CPU only test of SU(3) accuracy, tests 8 and 12 component reconstruction
void SU3Test() {

  Precision gauge_precision = QUDA_DOUBLE_PRECISION;
  double *gauge[4];



  QudaGaugeParam param;
  gauge_param = &param;

  param.X[0] = 4;
  param.X[1] = 4;
  param.X[2] = 4;
  param.X[3] = 4;
  setDims(param.X);

  // construct input fields
  for (int dir = 0; dir < 4; dir++) {
      gauge[dir] = (double*)malloc(V*gaugeSiteSize*sizeof(double));
      if(gauge[dir] == NULL){
	  printf("ERROR: malloc failed for gauge field\n");
	  exit(1);
      }
}
#define printVector(v)							\
	printf("{(%f %f) (%f %f) (%f %f)}\n", (v)[0], (v)[1], (v)[2], (v)[3], (v)[4], (v)[5]);

  param.anisotropy = 2.3;
  param.t_boundary = QUDA_ANTI_PERIODIC_T;
  param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  
  printf("Randomizing fields...");
  construct_gauge_field((void**)gauge, 1, gauge_precision);
  printf("done.\n");
  
  int fail_check = 17;
  int fail8[fail_check], fail12[fail_check];
  for (int f=0; f<fail_check; f++) {
      fail8[f] = 0;
    fail12[f] = 0;
  }

  int iter8[18], iter12[18];
  for (int i=0; i<18; i++) {
    iter8[i] = 0;
    iter12[i] = 0;
  }

  for (int eo=0; eo<2; eo++) {
      for (int i=0; i<Vh; i++) {
      int ga_idx = (eo*Vh+i);
      for (int d=0; d<4; d++) {
	double gauge8[18], gauge12[18];
	for (int j=0; j<18; j++) {
	  gauge8[j] = gauge[d][ga_idx*18+j];
	  gauge12[j] = gauge[d][ga_idx*18+j];
	}
	

	su3_construct(gauge8, QUDA_RECONSTRUCT_8, gauge_precision);
	su3_reconstruct(gauge8, d, i, QUDA_RECONSTRUCT_8, gauge_precision, eo);
	
	su3_construct(gauge12, QUDA_RECONSTRUCT_12, gauge_precision);
	su3_reconstruct(gauge12, d, i, QUDA_RECONSTRUCT_12, gauge_precision, eo);
	
	int err = 0;
	for (int ci = 0;ci < 18; ci++){
	    if (fabs(gauge12[ci] - gauge[d][ga_idx*18+ci]) > 1e-1
		|| fabs(gauge8[ci] - gauge[d][ga_idx*18+ci]) > 1e-1
		|| gauge12[ci] != gauge12[ci]
		|| gauge8[ci] != gauge8[ci]) {
		printf("index=%d\n", ci);
		err = 1;
		break;
	    }
	}
	
	if (err){
	
	    printf("original link\n");
	    printVector(gauge[d]+ga_idx*18);
	    printVector(gauge[d]+ga_idx*18+6);
	    printVector(gauge[d]+ga_idx*18+12);
	    printf("\n from 8-reconstruct\n");
	    printVector(&gauge8[0]);
	    printVector(&gauge8[6]);
	    printVector(&gauge8[12]);
	    printf("\n from 12-reconstruct\n");
	    printVector(&gauge12[0]);
	    printVector(&gauge12[6]);
	    printVector(&gauge12[12]);
	    printf("eo=%d, i=%d, d=%d\n", eo, i, d);
	    printf("Vh=%d\n", Vh);
	    exit(0);
	}

	for (int j=0; j<18; j++) {
	  double diff8 = fabs(gauge8[j] - gauge[d][ga_idx*18+j]);
	  double diff12 = fabs(gauge12[j] - gauge[d][ga_idx*18+j]);
	  for (int f=0; f<fail_check; f++) {
	    if (diff8 > pow(10,-(f+1))) fail8[f]++;
	    if (diff12 > pow(10,-(f+1))) fail12[f]++;
	  }
	  if (diff8 > 1e-3) {
	    iter8[j]++;
	  }
	  if (diff12 > 1e-3) {
	    iter12[j]++;
	  }

	}
      }
    }
  }

  for (int i=0; i<18; i++) printf("%d 12 fails = %d, 8 fails = %d\n", i, iter12[i], iter8[i]);

  for (int f=0; f<fail_check; f++) {
    printf("%e Failures: 12 component = %d / %d  = %e, 8 component = %d / %d = %e\n", 
	   pow(10,-(f+1)), fail12[f], V*4*18, fail12[f] / (double)(4*V*18),
	   fail8[f], V*4*18, fail8[f] / (double)(4*V*18));
  }

  // release memory
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);
}

int main(int argc, char **argv) {
  SU3Test();
}
