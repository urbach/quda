#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <typeinfo>
#include <quda.h>
#include <gauge_quda.h>
#include <quda_internal.h>
#include <face_quda.h>
#include "misc_helpers.h"
#include <assert.h>
#include <cuda.h>

#ifdef MPI_COMMS
#include "face_quda.h"
#endif

static double Anisotropy;
static QudaTboundary tBoundary;
static int X[4];
static int faceVolumeCB[4]; // checkboarded face volume
static int volumeCB; // checkboarded volume
extern float fat_link_max;

#include <pack_gauge.h>

template <typename Float>
void packGhost(Float **cpuLink, Float **cpuGhost, int nFace) {
  int XY=X[0]*X[1];
  int XYZ=X[0]*X[1]*X[2];

  //loop variables: a, b, c with a the most signifcant and c the least significant
  //A, B, C the maximum value
  //we need to loop in d as well, d's vlaue dims[dir]-3, dims[dir]-2, dims[dir]-1
  int A[4], B[4], C[4];
  
  //X dimension
  A[0] = X[3]; B[0] = X[2]; C[0] = X[1];
  
  //Y dimension
  A[1] = X[3]; B[1] = X[2]; C[1] = X[0];

  //Z dimension
  A[2] = X[3]; B[2] = X[1]; C[2] = X[0];

  //T dimension
  A[3] = X[2]; B[3] = X[1]; C[3] = X[0];


  //multiplication factor to compute index in original cpu memory
  int f[4][4]={
    {XYZ,    XY, X[0],     1},
    {XYZ,    XY,    1,  X[0]},
    {XYZ,  X[0],    1,    XY},
    { XY,  X[0],    1,   XYZ}
  };

  for(int dir =0; dir < 4; dir++)
    {
      Float* even_src = cpuLink[dir];
      Float* odd_src = cpuLink[dir] + volumeCB*gaugeSiteSize;

      Float* even_dst;
      Float* odd_dst;
     
     //switching odd and even ghost cpuLink when that dimension size is odd
     //only switch if X[dir] is odd and the gridsize in that dimension is greater than 1
      if((X[dir] % 2 ==0) || (commDim(dir) == 1)){
        even_dst = cpuGhost[dir];
        odd_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;	
     }else{
	even_dst = cpuGhost[dir] + nFace*faceVolumeCB[dir]*gaugeSiteSize;
        odd_dst = cpuGhost[dir];
     }

      int even_dst_index = 0;
      int odd_dst_index = 0;

      int d;
      int a,b,c;
      for(d = X[dir]- nFace; d < X[dir]; d++){
        for(a = 0; a < A[dir]; a++){
          for(b = 0; b < B[dir]; b++){
            for(c = 0; c < C[dir]; c++){
              int index = ( a*f[dir][0] + b*f[dir][1]+ c*f[dir][2] + d*f[dir][3])>> 1;
              int oddness = (a+b+c+d)%2;
              if (oddness == 0){ //even
                for(int i=0;i < 18;i++){
                  even_dst[18*even_dst_index+i] = even_src[18*index + i];
                }
                even_dst_index++;
              }else{ //odd
                for(int i=0;i < 18;i++){
                  odd_dst[18*odd_dst_index+i] = odd_src[18*index + i];
                }
                odd_dst_index++;
              }
            }//c
          }//b
        }//a
      }//d

      assert( even_dst_index == nFace*faceVolumeCB[dir]);
      assert( odd_dst_index == nFace*faceVolumeCB[dir]);
    }

}



void set_dim(int *XX) {

  volumeCB = 1;
  for (int i=0; i<4; i++) {
    X[i] = XX[i];
    volumeCB *= X[i];
    faceVolumeCB[i] = 1;
    for (int j=0; j<4; j++) {
      if (i==j) continue;
      faceVolumeCB[i] *= XX[j];
    }
    faceVolumeCB[i] /= 2;
  }
  volumeCB /= 2;

}

void pack_ghost(void **cpuLink, void **cpuGhost, int nFace, QudaPrecision precision) {

  if (precision == QUDA_DOUBLE_PRECISION) {
    packGhost((double**)cpuLink, (double**)cpuGhost, nFace);
  } else {
    packGhost((float**)cpuLink, (float**)cpuGhost, nFace);
  }

}



static void allocateGaugeField(FullGauge *cudaGauge, ReconstructType reconstruct, QudaPrecision precision) {

  cudaGauge->reconstruct = reconstruct;
  cudaGauge->precision = precision;

  cudaGauge->Nc = 3;

  int floatSize;
  if (precision == QUDA_DOUBLE_PRECISION) floatSize = sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) floatSize = sizeof(float);
  else floatSize = sizeof(float)/2;

  if (cudaGauge->even || cudaGauge->odd){
    errorQuda("Error: even/odd field is not null, probably already allocated(even=%p, odd=%p)\n", cudaGauge->even, cudaGauge->odd);
  }
 
  cudaGauge->bytes = 4*cudaGauge->stride*reconstruct*floatSize;
  if (!cudaGauge->even) {
    if (cudaMalloc((void **)&cudaGauge->even, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating even gauge field");
    }
  }
   
  if (!cudaGauge->odd) {
    if (cudaMalloc((void **)&cudaGauge->odd, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating even odd gauge field");
    }
  }

}


void freeGaugeField(FullGauge *cudaGauge) {
  if (cudaGauge->even) cudaFree(cudaGauge->even);
  if (cudaGauge->odd) cudaFree(cudaGauge->odd);
  cudaGauge->even = NULL;
  cudaGauge->odd = NULL;
}

template <typename Float, typename FloatN>
static void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, 
			   GaugeFieldOrder gauge_order, ReconstructType reconstruct, 
			   int bytes, int Vh, int pad, QudaLinkType type) {
  

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);

    

  if( ! packedEven ) errorQuda( "packedEven is borked\n");
  if( ! packedOdd ) errorQuda( "packedOdd is borked\n");
  if( ! even ) errorQuda( "even is borked\n");
  if( ! odd ) errorQuda( "odd is borked\n");
  if( ! cpuGauge ) errorQuda( "cpuGauge is borked\n");

#ifdef MULTI_GPU
  if (gauge_order != QUDA_QDP_GAUGE_ORDER)
    errorQuda("Only QUDA_QDP_GAUGE_ORDER is supported for multi-gpu\n");
#endif

  //for QUDA_ASQTAD_FAT_LINKS, need to find out the max value
  //fat_link_max will be used in encoding half precision fat link
  if(type == QUDA_ASQTAD_FAT_LINKS){
    for(int dir=0; dir < 4; dir++){
      for(int i=0;i < 2*Vh*gaugeSiteSize; i++){
	Float** tmp = (Float**)cpuGauge;
	if( tmp[dir][i] > fat_link_max ){
	  fat_link_max = tmp[dir][i];
	}
      }
    }
  }
  
  double fat_link_max_double = fat_link_max;
#ifdef MULTI_GPU
  reduceMaxDouble(fat_link_max_double);
#endif
  fat_link_max = fat_link_max_double;

  int voxels[] = {Vh, Vh, Vh, Vh};
  int nFace = 1;

  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    packQDPGaugeField(packedEven, (Float**)cpuGauge, 0, reconstruct, Vh, 
		      voxels, pad, 0, nFace, type);
    packQDPGaugeField(packedOdd,  (Float**)cpuGauge, 1, reconstruct, Vh, 
		      voxels, pad, 0, nFace, type);
  } else if (gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    packCPSGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh, pad);
    packCPSGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh, pad);    
  } else {
    errorQuda("Invalid gauge_order");
  }



#ifdef MULTI_GPU
#if 1
  // three step approach
  // 1. get the links into a contiguous buffer
  // 2. communicate between nodes
  // 3. pack into padded regions

  if (type == QUDA_ASQTAD_LONG_LINKS) nFace = 3;
  else nFace = 1;

  Float *ghostLink[4]; // ghost zone links
  Float *sendLink[4]; // send buffer for communication
  for(int i=0;i < 4;i++){
    ghostLink[i] = new Float[2*nFace*faceVolumeCB[i]*gaugeSiteSize*sizeof(Float)];
    if(ghostLink[i] == NULL) errorQuda("malloc failed for ghostLink[%d]\n", i);
    sendLink[i] = new Float[2*nFace*faceVolumeCB[i]*gaugeSiteSize*sizeof(Float)];
    if(sendLink[i] == NULL) errorQuda("malloc failed for sendLink[%d]\n", i);
  }

  packGhost((Float**)cpuGauge, sendLink, nFace); // pack the ghost zones into a contiguous buffer

  FaceBuffer face(X, 4, 18, nFace, QudaPrecision(sizeof(Float))); // this is the precision of the CPU field
  face.exchangeCpuLink((void**)ghostLink, (void**)sendLink);

  packQDPGaugeField(packedEven, ghostLink, 0, reconstruct, Vh,
		    faceVolumeCB, pad, Vh, nFace, type);
  packQDPGaugeField(packedOdd,  ghostLink, 1, reconstruct, Vh, 
		    faceVolumeCB, pad, Vh, nFace, type);
  
  for(int i=0;i < 4;i++) {
    delete []ghostLink[i];
    delete []sendLink[i];
  }
#else
  // Old QMP T-split code
  QudaPrecision precision = (QudaPrecision) sizeof(even->x);
  int Nvec = sizeof(FloatN)/precision;

  // one step approach
  // transfers into the pads directly
  transferGaugeFaces((void *)packedEven, (void *)(packedEven + Vh), precision, Nvec, reconstruct, Vh, pad);
  transferGaugeFaces((void *)packedOdd, (void *)(packedOdd + Vh), precision, Nvec, reconstruct, Vh, pad);
#endif
#endif
  checkCudaError();
  cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
    
  cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);
  checkCudaError();
  
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}


template <typename Float, typename FloatN>
static void retrieveGaugeField(Float *cpuGauge, FloatN *even, FloatN *odd, GaugeFieldOrder gauge_order,
			       ReconstructType reconstruct, int bytes, int Vh, int pad) {

  // Use pinned memory
  FloatN *packedEven, *packedOdd;
    
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);
    
  cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);    
    
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    unpackQDPGaugeField((Float**)cpuGauge, packedEven, 0, reconstruct, Vh, pad);
    unpackQDPGaugeField((Float**)cpuGauge, packedOdd, 1, reconstruct, Vh, pad);
  } else if (gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
    unpackCPSGaugeField((Float*)cpuGauge, packedEven, 0, reconstruct, Vh, pad);
    unpackCPSGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, Vh, pad);
  } else {
    errorQuda("Invalid gauge_order");
  }
    
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
}

void createGaugeField(FullGauge *cudaGauge, void *cpuGauge, QudaPrecision cuda_prec, QudaPrecision cpu_prec,
		      GaugeFieldOrder gauge_order, ReconstructType reconstruct, GaugeFixed gauge_fixed,
		      Tboundary t_boundary, int *XX, double anisotropy, double tadpole_coeff, int pad, QudaLinkType type)
{

  if (cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }

  Anisotropy = anisotropy;
  tBoundary = t_boundary;

  cudaGauge->anisotropy = anisotropy;
  cudaGauge->tadpole_coeff = tadpole_coeff;
  cudaGauge->volumeCB = 1;
  for (int d=0; d<4; d++) {
    cudaGauge->X[d] = XX[d];
    cudaGauge->volumeCB *= XX[d];
    X[d] = XX[d];
  }
  //cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
  cudaGauge->volumeCB /= 2;
  cudaGauge->pad = pad;

  /* test disabled because of staggered pad won't pass
#ifdef MULTI_GPU
  if (pad != cudaGauge->X[0]*cudaGauge->X[1]*cudaGauge->X[2]) {
    errorQuda("Gauge padding must match spatial volume");
  }
#endif
  */

  cudaGauge->stride = cudaGauge->volumeCB + cudaGauge->pad;
  cudaGauge->gauge_fixed = gauge_fixed;
  cudaGauge->t_boundary = t_boundary;
  
  allocateGaugeField(cudaGauge, reconstruct, cuda_prec);

  set_dim(XX);

  if (cuda_prec == QUDA_DOUBLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
		     gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (float*)cpuGauge, 
		     gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
    }

  } else if (cuda_prec == QUDA_SINGLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      } else {
	loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      }
    }

  } else if (cuda_prec == QUDA_HALF_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION){
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (double*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (reconstruct == QUDA_RECONSTRUCT_NO) {
	loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);
      } else {
	loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (float*)cpuGauge, 
		       gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, pad, type);	      
      }
    }

  }

}




void restoreGaugeField(void *cpuGauge, FullGauge *cudaGauge, QudaPrecision cpu_prec, GaugeFieldOrder gauge_order)
{
  if (cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }

  if (cudaGauge->precision == QUDA_DOUBLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      retrieveGaugeField((double*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      retrieveGaugeField((float*)cpuGauge, (double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), 
			 gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
    }

  } else if (cudaGauge->precision == QUDA_SINGLE_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpuGauge, (float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((double*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpuGauge, (float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((float*)cpuGauge, (float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    }

  } else if (cudaGauge->precision == QUDA_HALF_PRECISION) {

    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((double*)cpuGauge, (short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((double*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
      if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO) {
	retrieveGaugeField((float*)cpuGauge, (short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      } else {
	retrieveGaugeField((float*)cpuGauge, (short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), 
			   gauge_order, cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volumeCB, cudaGauge->pad);
      }
    }

  }

}



// The following routines are needed to test the hisq fermion force code
namespace hisq{
  namespace fermion_force{

    static void pack18Oprod(float2 *res, float *g, int dir, int Vh){
      float2 *r = res + dir*9*Vh;
      for(int j=0; j<9; j++){
        r[j*Vh].x = g[j*2+0];
        r[j*Vh].y = g[j*2+1];	
      }
      // r[dir][j][i] // where i is the position index, j labels segments of the su3 matrix, dir is direction
    }


    static void packOprodField(float2 *res, float *oprod, int oddBit, int Vh){
      // gaugeSiteSize = 18
      int gss=18;
      int total_volume = Vh*2;
      for(int dir=0; dir<8; dir++){
        float *op = oprod + dir*total_volume*gss + oddBit*Vh*gss; // prod[dir][0/1][x]
        for (int i=0; i<Vh; i++){
          pack18Oprod(res+i, op+i*gss, dir, Vh);
        }	
      }
    }

    static void packOprodFieldDir(float2 *res, float *oprod, int dir, int oddBit, int Vh){
      int gss=18;
      int total_volume = Vh*2;
      float *op = oprod + dir*total_volume*gss + oddBit*Vh*gss;
      for(int i=0; i<Vh; i++){
        pack18Oprod(res+i, op+i*gss, 0, Vh);
      }
    }


    static ParityOprod
      allocateParityOprod(int *X, QudaPrecision precision)
      {
        if(precision == QUDA_DOUBLE_PRECISION) {
          printf("Error: double precision not supported for compressed matrix\n");
          exit(0);
        }

        ParityOprod ret;

        ret.precision = precision;
        ret.X[0] = X[0]/2;
        ret.volume = X[0]/2;
        for(int d=1; d<4; d++){
          ret.X[d] = X[d];
          ret.volume *= X[d];
        }
        ret.Nc = 3;
        ret.length = ret.volume*ret.Nc*ret.Nc*2; // => number of real numbers       

        if(precision == QUDA_DOUBLE_PRECISION){
          ret.bytes = ret.length*sizeof(double);
        }
        else { // QUDA_SINGLE_PRECISION
          ret.bytes = ret.length*sizeof(float);
        }

        // loop over the eight directions 
        for(int dir=0; dir<8; dir++){
          if(cudaMalloc((void**)&(ret.data[dir]), ret.bytes) == cudaErrorMemoryAllocation){
            printf("Error allocating ParityOprod\n");
            exit(0);
          }
          cudaMemset(ret.data[dir], 0, ret.bytes);
        }

        return ret;
      }



    FullOprod
      createOprodQuda(int *X, QudaPrecision precision)
      {
        FullOprod ret;
        ret.even = allocateParityOprod(X, precision);
        ret.odd  = allocateParityOprod(X, precision);
        return ret;
      }




    static void
      copyOprodFromCPUArrayQuda(FullOprod cudaOprod, void *cpuOprod,
          size_t bytes_per_dir, int Vh)
      {
        // Use pinned memory 
        float2 *packedEven, *packedOdd;
        cudaMallocHost((void**)&packedEven, bytes_per_dir); // now
        cudaMallocHost((void**)&packedOdd, bytes_per_dir);
        for(int dir=0; dir<8; dir++){
          packOprodFieldDir(packedEven, (float*)cpuOprod, dir, 0, Vh);
          packOprodFieldDir(packedOdd,  (float*)cpuOprod, dir, 1, Vh);

          cudaMemset(cudaOprod.even.data[dir], 0, bytes_per_dir);
          cudaMemset(cudaOprod.odd.data[dir],  0, bytes_per_dir);
          checkCudaError();

          cudaMemcpy(cudaOprod.even.data[dir], packedEven, bytes_per_dir, cudaMemcpyHostToDevice);
          cudaMemcpy(cudaOprod.odd.data[dir], packedOdd, bytes_per_dir, cudaMemcpyHostToDevice);
          checkCudaError();
        }
        cudaFreeHost(packedEven);
        cudaFreeHost(packedOdd);
      }


    void copyOprodToGPU(FullOprod cudaOprod,
        void  *oprod,
        int half_volume){
      int bytes_per_dir = half_volume*18*sizeof(float);
      copyOprodFromCPUArrayQuda(cudaOprod, oprod, bytes_per_dir, half_volume);
    }




    static void unpack18Oprod(float *h_oprod, float2 *d_oprod, int dir, int Vh) {
      float2 *dg = d_oprod + dir*9*Vh;
      for (int j=0; j<9; j++) {
        h_oprod[j*2+0] = dg[j*Vh].x; 
        h_oprod[j*2+1] = dg[j*Vh].y;
      }
    }

    static void unpackOprodField(float* res, float2* cudaOprod, int oddBit, int Vh){
      int gss=18; 
      int total_volume = Vh*2;
      for(int dir=0; dir<8; dir++){
        float* res_ptr = res + dir*total_volume*gss + oddBit*Vh*gss;
        for(int i=0; i<Vh; i++){
          unpack18Oprod(res_ptr + i*gss, cudaOprod+i, dir, Vh);
        }
      }
    }


    static void 
      fetchOprodFromGPUArraysQuda(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod, size_t bytes, int Vh)
      {
        float2 *packedEven, *packedOdd;
        cudaMallocHost((void**)&packedEven,bytes);
        cudaMallocHost((void**)&packedOdd, bytes);


        cudaMemcpy(packedEven, cudaOprodEven, bytes, cudaMemcpyDeviceToHost);
        checkCudaError();
        cudaMemcpy(packedOdd, cudaOprodOdd, bytes, cudaMemcpyDeviceToHost);
        checkCudaError();

        unpackOprodField((float*)cpuOprod, packedEven, 0, Vh);
        unpackOprodField((float*)cpuOprod, packedOdd,  1, Vh);

        cudaFreeHost(packedEven);
        cudaFreeHost(packedOdd);
      }

    void fetchOprodFromGPU(void *cudaOprodEven,
        void *cudaOprodOdd,
        void *oprod,
        int half_vol){
      int bytes = 8*half_vol*18*sizeof(float);
      fetchOprodFromGPUArraysQuda(cudaOprodEven, cudaOprodOdd, oprod, bytes, half_vol);
    }

    static void 
      loadOprodFromCPUArrayQuda(void *cudaOprodEven, void *cudaOprodOdd, void *cpuOprod,
          size_t bytes, int Vh)
      {
        // Use pinned memory 
        float2 *packedEven, *packedOdd;
        cudaMallocHost((void**)&packedEven, bytes);
        cudaMallocHost((void**)&packedOdd, bytes);


        packOprodField(packedEven, (float*)cpuOprod, 0, Vh);
        packOprodField(packedOdd,  (float*)cpuOprod, 1, Vh);

        cudaMemset(cudaOprodEven, 0, bytes);
        cudaMemset(cudaOprodOdd, 0, bytes);
        checkCudaError();

        cudaMemcpy(cudaOprodEven, packedEven, bytes, cudaMemcpyHostToDevice);
        checkCudaError();
        cudaMemcpy(cudaOprodOdd, packedOdd, bytes, cudaMemcpyHostToDevice);
        checkCudaError();

        cudaFreeHost(packedEven);
        cudaFreeHost(packedOdd);
      }



    void loadOprodToGPU(void *cudaOprodEven, 
        void *cudaOprodOdd,
        void  *oprod,
        int vol){
      int bytes = 8*vol*18*sizeof(float);
      loadOprodFromCPUArrayQuda(cudaOprodEven, cudaOprodOdd, oprod, bytes, vol);
    }



    void allocateOprodFields(void **cudaOprodEven, void **cudaOprodOdd, int vol){
      int bytes = 8*vol*18*sizeof(float);

      if (cudaMalloc((void **)cudaOprodEven, bytes) == cudaErrorMemoryAllocation) {
        errorQuda("Error allocating even outer product field");
      }

      cudaMemset((*cudaOprodEven), 0, bytes);

      checkCudaError();

      if (cudaMalloc((void **)cudaOprodOdd, bytes) == cudaErrorMemoryAllocation) {
        errorQuda("Error allocating odd outer product field");
      }

      cudaMemset((*cudaOprodOdd), 0, bytes);

      checkCudaError();
    }

  } // end namespace fermion_force
} // end namespace hisq
