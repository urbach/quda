#include <cuda.h>
#include <quda_internal.h> // Nstream is defined here
//#include <dslash_constants.h> // contains ghostFace
//#include <pack_face_def.h> // kernels for packing
#include <face_quda.h>     // faceVolumeCB is declared here
			   // and is set in FaceBuffer::setDims


// ghostFace is initialised in initLatticeConstants(const LatticeField &lat);
// And where/how is faceBuffer initialised?


// This file will contain the code needed to compute the quark-field 
// outer products needed for the fermion force, on the GPU.
// Textures have file scope
texture<float2, 1, cudaReadModeElementType> colorVecTexSingle;
// No texture for double2 types, so "fake" with int4
texture<int4, 1, cudaReadModeElementType> colorVecTexDouble;

#include <read_gauge.h>
#include <gauge_field.h>

namespace {
#include <float_vector.h>
}; // anonymous namespace - sloppy!!

#define sp_stride Vh
#ifndef BLOCK_DIM
#define BLOCK_DIM 64
#endif

namespace quda {

namespace fermion_force {

template<class RealA>
__device__
void printMatrix(const RealA* const mat){

  printf("(%lf %lf),  (%lf, %lf), (%lf,%lf)\n", mat[0].x, mat[0].y, mat[1].x, mat[1].y, mat[2].x, mat[2].y);
  printf("(%lf %lf),  (%lf, %lf), (%lf,%lf)\n", mat[3].x, mat[3].y, mat[4].x, mat[4].y, mat[5].x, mat[5].y);
  printf("(%lf %lf),  (%lf, %lf), (%lf,%lf)\n", mat[6].x, mat[6].y, mat[7].x, mat[7].y, mat[8].x, mat[8].y);

  return;
}

template<class RealA> 
__device__
void printVector(const RealA* const vec){
  printf("(%lf %lf),  (%lf, %lf), (%lf,%lf)\n", vec[0].x, vec[0].y, vec[1].x, vec[1].y, vec[2].x, vec[2].y);
}



template<class T>
inline __device__
void computeOuterProduct(const T* const u, 
			 const T* const v,
			 T* const  mat)
{
#define INDEX(a, b) a*3 + b
#pragma loop unroll
  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j){
      mat[INDEX(i,j)].x = u[i].x*v[j].x + u[i].y*v[j].y;
      mat[INDEX(i,j)].y = u[i].y*v[j].x - u[i].x*v[j].y;
    }
  } 
#undef INDEX
  return;
}


// Need to write routines that read in the quark fields
// I will use texture memory for this, I think.
/*
inline __device__  // Need to figure out what type field is
void readColorVectorFromTexture(int index, int stride, double2 vec[3])
{
  vec[0] = fetch_double2((colorVecTexDouble), index+0*stride);
  vec[1] = fetch_double2((colorVecTexDouble), index+1*stride);
  vec[2] = fetch_double2((colorVecTexDouble), index+2*stride);
  return;
}
*/

inline __device__
void readColorVectorFromTexture(int index, int stride, float2 vec[3])
{
  vec[0] = tex1Dfetch((colorVecTexSingle), index);
  vec[1] = tex1Dfetch((colorVecTexSingle), index + stride);
  vec[2] = tex1Dfetch((colorVecTexSingle), index + 2*stride);
  return;
}

template<class T>
inline __device__ 
void readColorVectorFromField(const T* const field, int index, int stride, T vec[3])
{
  vec[0] = *(field + index);
  vec[1] = *(field + index + stride);
  vec[2] = *(field + index + 2*stride);
  return;
}


// only works if Promote<T,U>::Type = T
// This is not what I want to do really.
template<class T, class U>
inline __device__
void addOprodMatrixToField(const T* const mat, int dir, int idx, U coeff, T* const field)
{
  field[idx + dir*Vh*9]          += coeff*mat[0];
  field[idx + dir*Vh*9 + Vh]     += coeff*mat[1];
  field[idx + dir*Vh*9 + Vh*2]   += coeff*mat[2];
  field[idx + dir*Vh*9 + Vh*3]   += coeff*mat[3];
  field[idx + dir*Vh*9 + Vh*4]   += coeff*mat[4];
  field[idx + dir*Vh*9 + Vh*5]   += coeff*mat[5];
  field[idx + dir*Vh*9 + Vh*6]   += coeff*mat[6];
  field[idx + dir*Vh*9 + Vh*7]   += coeff*mat[7];
  field[idx + dir*Vh*9 + Vh*8]   += coeff*mat[8];
  return;
}


template<class T, class U>
inline __device__
void loadOprodMatrixFromField(const T* const field, int dir, int idx, U coeff, T* const mat)
{
  mat[0] = field[idx + dir*Vh*9];
  mat[1] = field[idx + dir*Vh*9 + Vh];
  mat[2] = field[idx + dir*Vh*9 + Vh*2];
  mat[3] = field[idx + dir*Vh*9 + Vh*3];
  mat[4] = field[idx + dir*Vh*9 + Vh*4];
  mat[5] = field[idx + dir*Vh*9 + Vh*5];
  mat[6] = field[idx + dir*Vh*9 + Vh*6];
  mat[7] = field[idx + dir*Vh*9 + Vh*7];
  mat[8] = field[idx + dir*Vh*9 + Vh*8];
  return;
}



// First pass at the Multi-gpu build
#ifdef MULTI_GPU
struct OprodParam {
  int threads;
  int commDim[4];
  int ghostDim[4];
  int ghostOffset[4];
  KernelType kernelType;
};


#define Vsh_x ghostFace[0]
#define Vsh_y ghostFace[1]
#define Vsh_z ghostFace[2]
#define Vsh_t ghostFace[3]

// setup
void createOprodEvents()
{
  for(int i=0; i<Nstream; ++i){
    cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
    cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
  }
  checkCudaError();
  return; 
}

// cleanup
void destroyOprodEvents()
{
  for(int i=0; i<Nstream; ++i){
    cudaEventDestroy(packEnd[i]);
    cudaEventDestroy(gatherStart[i]);
    cudaEventDestroy(gatherEnd[i]);
    cudaEventDestroy(scatterStart[i]);
    cudaEventDestroy(scatterEnd[i]);
  }
  checkCudaError();
} 


// The kernel
template<class Real, class RealA, bool forNaik, int oddBit> 
__global__ 
void computeOuterProdKernel(const RealA* const thisField,  const RealA* const neighborField, Real coeff, OprodParam oprodParam, RealA* const oprodField)
{
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  if(sid >= oprodParam.threads) return;


  int za, zb;
  int x1h, x2h;
  int x1, x2, x3, x4;
  int x1_new, x2_new, x3_new, x4_new;
  int x1odd, x2odd;
  int X;
  int af;
  int old_sid = sid;

  if(oprodParam.kernelType == INTERIOR_KERNEL){

    if(sid==0){ printf("kernelType == INTERIOR_KERNEL\n"); }
    //data order: X4 X3 X2 X1h
    za = sid / X1h;
    x1h = sid - za*X1h;
    zb = za / X2;
    x2 = za - zb*X2;
    x4 = zb / X3;
    x3 = zb - x4*X3;
    x1odd = (x2 + x3 + x4 + oddBit) & 1;
    x1 = 2*x1h + x1odd;
    X = 2*sid + x1odd;
  }else if (oprodParam.kernelType == EXTERIOR_KERNEL_X){
  
   if(sid==0){ printf("kernelType == EXTERIOR_KERNEL_X\n"); }
    //data order: X1 X4 X3 X2h
    za = sid / X2h;
    x2h = sid - za*X2h;
    zb = za / X3;
    x3 = za - zb*X3;
    x1 = zb / X4;
    x4 = zb - x1*X4;
    af = (x1 >= 3)?(X1-6):0;
    x1_new = x1 + af;
    x1=x1_new;
    x2odd = (x3 + x4 + x1 + oddBit) & 1;
    x2 = 2*x2h + x2odd;
    X = x4*X3X2X1+x3*X2X1+x2*X1+x1;
    sid = X>>1;
  }else if (oprodParam.kernelType == EXTERIOR_KERNEL_Y){
   
    if(sid==0){ printf("kernelType == EXTERIOR_KERNEL_Y\n"); }
    //data order: X2 X4 X3 X1h
    za = sid / X1h;
    x1h = sid - za*X1h;
    zb = za / X3;
    x3 = za - zb*X3;
    x2 = zb / X4;
    x4 = zb - x2*X4;
    af = (x2 >= 3)?(X2-6):0;
    x2_new = x2 + af;
    x2=x2_new;
    x1odd = (x3 + x4 + x2 + oddBit) & 1;
    x1 = 2*x1h + x1odd;
    X = x4*X3X2X1+x3*X2X1+x2*X1+x1;
    sid = X>>1;
  }else if (oprodParam.kernelType == EXTERIOR_KERNEL_Z){
    if(sid==0){ printf("kernelType == EXTERIOR_KERNEL_Z\n"); }
    //data order: X3 X4 X2 X1h
    za = sid / X1h;
    x1h = sid - za*X1h;
    zb = za / X2;
    x2 = za - zb*X2;
    x3 = zb / X4;
    x4 = zb - x3*X4;
    af = (x3 >= 3)?(X3-6):0;
    x3_new = x3 + af;
    x3=x3_new;
    x1odd = (x2 + x4 + x3 + oddBit) & 1;
    x1 = 2*x1h + x1odd;
    X = x4*X3X2X1+x3*X2X1+x2*X1+x1;
    sid = X>>1;
   }else if (oprodParam.kernelType == EXTERIOR_KERNEL_T){
    //data order: X4 X3 X2 X1h
    za = sid / X1h;
    x1h = sid - za*X1h;
    zb = za / X2;
    x2 = za - zb*X2;
    x4 = zb / X3;
    x3 = zb - x4*X3;


    af = (x4 >= 3)?(X4-6):0;
    x4_new = x4 + af;
    sid +=Vsh*(x4_new -x4);
    x4=x4_new;
    x1odd = (x2 + x3 + x4 + oddBit) & 1;
    x1 = 2*x1h + x1odd;
    X = 2*sid + x1odd;
   }


  RealA this_vector[3];     // colorVector at sid
  RealA neighbor_vector[3]; // colorVector at neighboring site
  RealA oprod[9];           // outer product

  int neighbor_index;
  int stride;

  // read the color vector at this lattice site
  // we only run on even half lattice sites
  // need to set stride!!

  // hop in positive x direction

 if(forNaik){
  if( oprodParam.kernelType == EXTERIOR_KERNEL_X && (x1 >= X1m3) ){
    stride = 3*Vsh_x;
  }else if( oprodParam.kernelType == EXTERIOR_KERNEL_Y && (x2 >= X2m3) ){
    stride = 3*Vsh_y;
  }else if( oprodParam.kernelType == EXTERIOR_KERNEL_Z && (x3 >= X3m3) ){
    stride = 3*Vsh_z;
  }else if( oprodParam.kernelType == EXTERIOR_KERNEL_T && (x4 >= X4m3) ){
    stride = 3*Vsh_t;
  }else{
    stride = Vh;
  }
  readColorVectorFromField(thisField, sid, Vh, this_vector);

  if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[0] || x1 < X1m3) ) ||
        oprodParam.kernelType == EXTERIOR_KERNEL_X && (x1 >= X1m3) ){
     if((oprodParam.kernelType == EXTERIOR_KERNEL_X) && (x1 >= X1m3)){
        int space_con = (x4*X3X2 + x3*X2 + x2)/2;
        neighbor_index = oprodParam.ghostOffset[0] + 9*Vsh_x + (x1-X1m3)*Vsh_x + space_con;
      }else{
        neighbor_index = ((x1 >= X1m3) ? X -X1m3 : X+3) >> 1;
      }
      readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
      computeOuterProduct(neighbor_vector, this_vector, oprod);
      addOprodMatrixToField(oprod, 0, sid, coeff, oprodField + oddBit*4*Vh*9); // offset
   } // x direction

   // hop in positive y direction
   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[1] || x2 < X2m3) ) ||
      oprodParam.kernelType == EXTERIOR_KERNEL_Y && (x2 >= X2m3) ){
      if((oprodParam.kernelType == EXTERIOR_KERNEL_Y) && (x2 >= X2m3)){
        int space_con = (x4*X3X1 + x3*X1 + x1)/2;
        neighbor_index = oprodParam.ghostOffset[1] + 9*Vsh_y + (x2-X2m3)*Vsh_y + space_con;
      }else{
        neighbor_index = ((x2 >= X2m3 ) ? X-X2m3*X1 : X+3*X1) >> 1; 
      }
      readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
      computeOuterProduct(neighbor_vector, this_vector, oprod);
      addOprodMatrixToField(oprod, 1, sid, coeff, oprodField + oddBit*4*Vh*9); // offset + change of sign  
   } // y direction

    // hop in positive z direction
   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[2] || x3 < X3m3) ) ||
     oprodParam.kernelType == EXTERIOR_KERNEL_Z && (x3 >= X3m3) ){
     if((oprodParam.kernelType == EXTERIOR_KERNEL_Z) && (x3 >= X3m3) ){
       int space_con = (x4*X2X1+x2*X1+x1)/2;
       neighbor_index = oprodParam.ghostOffset[2] + 9*Vsh_z +(x3-X3m3)*Vsh_z + space_con;
     }else{
       neighbor_index = ((x3>= X3m3)? X -X3m3*X2X1: X + 3*X2X1)>> 1;
     }
     readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
     computeOuterProduct(neighbor_vector, this_vector, oprod);
     addOprodMatrixToField(oprod, 2, sid, coeff, oprodField + oddBit*4*Vh*9); // offset + change of sign  
   } // z direction

   // hop in positive t direction
   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[3] || x4 < X4m3) ) ||
     oprodParam.kernelType == EXTERIOR_KERNEL_T && (x4 >= X4m3) ){
     if((oprodParam.kernelType == EXTERIOR_KERNEL_T) && (x4 >= X4m3) ){
       int space_con = (x3*X2X1+x2*X1+x1)/2;
       neighbor_index = oprodParam.ghostOffset[3] + 9*Vsh_t +(x4-X4m3)*Vsh_t+ space_con;
     }else{
       neighbor_index = ((x4>=X4m3)? X -X4m3*X3X2X1 : X + 3*X3X2X1)>> 1; 
     }
     readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
     computeOuterProduct(neighbor_vector, this_vector, oprod);
     addOprodMatrixToField(oprod, 3, sid, coeff, oprodField + oddBit*4*Vh*9); // offset + change of sign  
   } // t direction

 }else{ // !forNaik

   if( oprodParam.kernelType == EXTERIOR_KERNEL_X && (x1 >= X1m1) ){
     stride = 3*Vsh_x;
   }else if( oprodParam.kernelType == EXTERIOR_KERNEL_Y && (x2 >= X2m1) ){
     stride = 3*Vsh_y;
   }else if( oprodParam.kernelType == EXTERIOR_KERNEL_Z && (x3 >= X3m1) ){
     stride = 3*Vsh_z;
   }else if( oprodParam.kernelType == EXTERIOR_KERNEL_T && (x4 >= X4m1) ){
     stride = 3*Vsh_t;
   }else{
     stride = Vh;
   }
   readColorVectorFromField(thisField, sid, Vh, this_vector);


   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[0] || x1 < X1m1) ) ||
        oprodParam.kernelType == EXTERIOR_KERNEL_X && (x1 >= X1m1) ){
     if((oprodParam.kernelType == EXTERIOR_KERNEL_X) && (x1 >= X1m1)){
        int space_con = (x4*X3X2 + x3*X2 + x2)/2;
        neighbor_index = oprodParam.ghostOffset[0] + 9*Vsh_x + (x1-X1m1)*Vsh_x + space_con;
      }else{
        neighbor_index = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
      }
      readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
      computeOuterProduct(neighbor_vector, this_vector, oprod);
      addOprodMatrixToField(oprod, 0, sid, coeff, oprodField + oddBit*4*Vh*9); // offset
   } // x direction



   // hop in positive y direction
   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[1] || x2 < X2m1) ) ||
      oprodParam.kernelType == EXTERIOR_KERNEL_Y && (x2 >= X2m1) ){
      if((oprodParam.kernelType == EXTERIOR_KERNEL_Y) && (x2 >= X2m1)){
        int space_con = (x4*X3X1 + x3*X1 + x1)/2;
        neighbor_index = oprodParam.ghostOffset[1] + 9*Vsh_y + (x2-X2m1)*Vsh_y + space_con;
      }else{
        neighbor_index = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
      }
      readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
      computeOuterProduct(neighbor_vector, this_vector, oprod);
      addOprodMatrixToField(oprod, 1, sid, coeff, oprodField + oddBit*4*Vh*9); // offset
   } // y direction



   // hop in positive z direction
   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[2] || x3 < X3m1) ) ||
     oprodParam.kernelType == EXTERIOR_KERNEL_Z && (x3 >= X3m1) ){
     if((oprodParam.kernelType == EXTERIOR_KERNEL_Z) && (x3 >= X3m1) ){
       int space_con = (x4*X2X1+x2*X1+x1)/2;
       neighbor_index = oprodParam.ghostOffset[2] + 9*Vsh_z +(x3-X3m1)*Vsh_z + space_con;
     }else{
       neighbor_index = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
     }
     readColorVectorFromField(neighborField, neighbor_index, stride, neighbor_vector);
     computeOuterProduct(neighbor_vector, this_vector, oprod);
     addOprodMatrixToField(oprod, 2, sid, coeff, oprodField + oddBit*4*Vh*9); // offset    

     if(oprodParam.kernelType == EXTERIOR_KERNEL_Z){
		  if(x1==1 && x2==0 && x3==5 && x4==0){
		     printf("(%d, %d, %d, %d) - Old index : %d, New index : %d, Neighbor Index : %d \n", x1, x2, x3, x4, old_sid, sid, neighbor_index);
		     printf("This Vec...\n");
		     printVector(this_vector);
		     printf("Neighbor Vec...\n");
		     printVector(neighbor_vector);
		     printf("Oprod...\n");
		     printMatrix(oprod);
	  	}
     }else{
	     if(x1==1 && x2==0 && x3==5 && x4==0){
	     //if(x1==1 && x2==0 && x3==5 && x4==0){
		     printf("(%d, %d, %d, %d) - , Index : %d, Neighbor Index : %d \n", x1, x2, x3, x4, sid, neighbor_index);
		     printf("This Vec...\n");
		     printVector(this_vector);
		     printf("Neighbor Vec...\n");
		     printVector(neighbor_vector);
		     printf("Oprod...\n");
		     printMatrix(oprod);
	     }
     }
   } // z direction



   // hop in positive t direction
   if( (oprodParam.kernelType == INTERIOR_KERNEL && (!oprodParam.ghostDim[3] || x4 < X4m1) ) ||
     oprodParam.kernelType == EXTERIOR_KERNEL_T && (x4 >= X4m1) ){
     if((oprodParam.kernelType == EXTERIOR_KERNEL_T) && (x4 >= X4m1) ){
       int space_con = (x3*X2X1+x2*X1+x1)/2;
       neighbor_index = oprodParam.ghostOffset[3] + 9*Vsh_t +(x4-X4m1)*Vsh_t+ space_con;
     }else{
       neighbor_index = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
     }
     readColorVectorFromField(neighborField,  neighbor_index, stride, neighbor_vector);
     computeOuterProduct(neighbor_vector, this_vector, oprod);
     addOprodMatrixToField(oprod, 3, sid, coeff, oprodField + oddBit*4*Vh*9); // offset  
   } // t direction
  } // !(forNaik)
  return;
}



// Replaces a whole bunch of global variables
struct CommPattern 
{
  int gatherCompleted[Nstream];
  int previousDir[Nstream];
  int commsCompleted[Nstream];
  int commDimTotal;
};


void initOprodCommsPattern(const OprodParam& oprodParam, CommPattern* commPattern) {
  for(int i=0; i<Nstream-1; ++i){
    commPattern->gatherCompleted[i] = 0; 
    commPattern->commsCompleted[i] = 0;
  }
  commPattern->gatherCompleted[Nstream-1] = 1;
  commPattern->commsCompleted[Nstream-1] = 1;
  // We need to know which was the previous direction in which 
  // communication was issued, since we only query a given event / 
  // comms call after the previous one has successfully completed

  for(int i=3; i>=0; i--){
    if(oprodParam.commDim[i]){
      int prev = Nstream-1;
      for(int j=3; j>i; j--) if(oprodParam.commDim[j]) prev = 2*j;
      commPattern->previousDir[2*i + 1] = prev;
      commPattern->previousDir[2*i + 0] = 2*i + 1; // always valid
    } // if(oprodParam.commDim[i])
  } // loop over i
  // commDim[3] == 1
  // then
  // previousDir[7] = Nstream-1 // I think that Nstream = 1
  // previousDir[6] = 7
  // previousDir[5] = 6
  // previousDir[4] = 5
  // previousDir[3] = 6
  // previousDir[2] = 3
  // previousDir[1] = 6
  // previousDir[0] = 1
   

  // This tells us how many events / comms occurrances there are in total
  // Used for exiting the while loop 
  commPattern->commDimTotal = 0;
  for(int i=3; i>=0; i--) commPattern->commDimTotal += oprodParam.commDim[i];
  commPattern->commDimTotal *= 4; // 2 from pipe length, 2 from direction 
		     // commDimTotal can range up to 16
  return;
}





template<int oddBit, bool forNaik> 
static void computeOprodKernelWrapper(cudaStream_t stream, 
			      const QudaGaugeParam& gaugeParam,
			      double coeff,
			      const  cudaColorSpinorField& thisField,
			      const  cudaColorSpinorField& neighborField,
			      const OprodParam& oprodParam,
			      cudaGaugeField* const oprodField)
{

  printfQuda("Calling computeOprodKernelWrapper\n");
  printfQuda("oprodParam.threads = %d\n", oprodParam.threads);

  dim3 blockDim(BLOCK_DIM,1,1);
  dim3 gridDim((oprodParam.threads+BLOCK_DIM-1)/BLOCK_DIM,1,1);

  if(gaugeParam.cuda_prec == QUDA_SINGLE_PRECISION){
    computeOuterProdKernel<float, float2, forNaik, oddBit><<<gridDim,blockDim, 0, stream>>>((const float2* const)thisField.V(), 
											    (const float2* const)neighborField.V(),
											    (float)coeff, oprodParam, (float2* const)oprodField->Gauge_p());
    checkCudaError();
  }else if(gaugeParam.cuda_prec == QUDA_DOUBLE_PRECISION){
    computeOuterProdKernel<double, double2, forNaik, oddBit><<<gridDim,blockDim, 0, stream>>>((const double2* const)thisField.V(), 
											      (const double2* const)neighborField.V(),
											       coeff, oprodParam, (double2* const)oprodField->Gauge_p());
    checkCudaError();
  }else{
    errorQuda("Unrecognised precision\n");
    exit(1);
  }
  cudaThreadSynchronize();
  return;
}



template<int oddBit, int forNaik>
void computeOuterProdCuda(const QudaGaugeParam& gaugeParam,
			  double coeff,
			  cudaColorSpinorField& thisField,	
			  cudaColorSpinorField& neighborField,	
			  cudaGaugeField* oprodField)
{
  const int Npad = (thisField.Ncolor()*thisField.Nspin()*2)/(thisField.FieldOrder());

  
  const int* const faceVolumeCB = thisField.GhostFace();
  printfQuda("faceVolumeCB = %d %d %d %d\n", faceVolumeCB[0],
  					 faceVolumeCB[1],
  					 faceVolumeCB[2],
  					 faceVolumeCB[3]);


  OprodParam oprodParam;
  const int commOverride[4] = {1,1,1,1};
  for(int dir=0; dir<4; ++dir){
    oprodParam.ghostDim[dir]        = commDimPartitioned(dir); // declared in face_quda.h
    oprodParam.ghostOffset[dir]     = Npad*(thisField.GhostOffset(dir) + thisField.Stride());
    oprodParam.commDim[dir]         = (!commOverride[dir]) ? 0 : commDimPartitioned(dir);
    printfQuda("oprodParam.ghostOffset[%d] = %d\n", dir, oprodParam.ghostOffset[dir]);
  }
  printfQuda("quarkField.Stride() == %d\n", thisField.Stride());

  oprodParam.kernelType = INTERIOR_KERNEL;
  oprodParam.threads = thisField.Volume(); 
  // For the exterior kernel only. Changes below.
  // need to set oprodParam.commDim, oprodParam.ghostDim, oprodParam.ghostOffset

  const int ndim  = 4;
  const int nface = 3; 
  const int num_quark_reals = 6;
  FaceBuffer faceBuffer(gaugeParam.X, ndim, num_quark_reals, nface, gaugeParam.cpu_prec);
  CommPattern commPattern;
  const int dagger = 0;

  for(int i = 3; i >=0; i--){
    if(oprodParam.commDim[i]){
      printfQuda("Partitioned in direction %d\n", i);
      faceBuffer.pack(neighborField, 1-oddBit, 0, i, streams);
      cudaEventRecord(packEnd[2*i+0], streams[Nstream-1]); 
      cudaEventRecord(packEnd[2*i+1], streams[Nstream-1]);
    } // param.commDim[i]
  } // i

  for(int i=3; i >=0; i--){
    if (oprodParam.commDim[i]){
      printfQuda("oprodParam.commDim[%d] is set\n", i);
      for (int dir=1; dir>=0; dir--) {
        cudaStreamWaitEvent(streams[2*i+dir], packEnd[2*i+dir], 0);
        faceBuffer.gather(neighborField, dagger, 2*i+dir);
        cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]);
      } // dir 
    } // param.commDim[i]
  } // i 
  // Call the interior kernel


  printfQuda("Calling computOprodKernelWrapper\n");
  computeOprodKernelWrapper<oddBit, forNaik>(streams[Nstream-1], gaugeParam, coeff, thisField, neighborField, oprodParam, oprodField); 
  checkCudaError();
  initOprodCommsPattern(oprodParam,&commPattern);

  int completeSum = 0;
  while(completeSum < commPattern.commDimTotal) {
    for(int i=3; i>=0; i--) {
      if(oprodParam.commDim[i]){
      
        for (int dir=1; dir>=0; dir--) {
	  // Query if gather has completed
	  if (!(commPattern.gatherCompleted[2*i+dir]) && commPattern.gatherCompleted[commPattern.previousDir[2*i+dir]]) { 
	    if (cudaSuccess == cudaEventQuery(gatherEnd[2*i+dir])) {
	      commPattern.gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      gettimeofday(&commsStart[2*i+dir], NULL);
	      faceBuffer.commsStart(2*i+dir);
	    }
	  }
	  // Query if comms has finished
	  if (!(commPattern.commsCompleted[2*i+dir]) && commPattern.commsCompleted[commPattern.previousDir[2*i+dir]] &&
	      commPattern.gatherCompleted[2*i+dir]) {
	    if (faceBuffer.commsQuery(2*i+dir)) { 
	      commPattern.commsCompleted[2*i+dir] = 1;
	      completeSum++;
	      gettimeofday(&commsEnd[2*i+dir], NULL);
	      faceBuffer.scatter(neighborField, dagger, 2*i+dir);
	      cudaEventRecord(scatterEnd[2*i+dir], streams[2*i+dir]);
	    }
	  }
        } // loop over dir
      } // if(oprodParam.dimComm[i])
    } // loop over i
  } // while(completeSum < commDimTotal)




  // Exterior kernels
  for (int i=3; i>=0; i--) {
    if (oprodParam.commDim[i]){ 
      oprodParam.kernelType = static_cast<KernelType>(i);
      oprodParam.threads = 6*faceVolumeCB[i]; // updating 2 or 6 faces
      // wait for scattering to finish and then launch dslash
      cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0);
      cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i+1], 0);
      computeOprodKernelWrapper<oddBit, forNaik>(streams[Nstream-1], gaugeParam, coeff, thisField, neighborField, oprodParam, oprodField); 
    } //(if param.commDim[i])
  } // loop over i

  return;
}



void computeOuterProdCuda(int oddBit, int forNaik,
			  const QudaGaugeParam& gaugeParam,
			  double coeff,
			  cudaColorSpinorField& thisField,	
			  cudaColorSpinorField& neighborField,	
			  cudaGaugeField* oprodField)
{
  if(forNaik){
    if(oddBit){
      computeOuterProdCuda<1,true>(gaugeParam, coeff, thisField, neighborField, oprodField);
    }else{
      computeOuterProdCuda<0,true>(gaugeParam, coeff, thisField, neighborField, oprodField);
    }
  }else{ // !forNaik
    if(oddBit){
      computeOuterProdCuda<1,false>(gaugeParam, coeff, thisField, neighborField, oprodField);
    }else{
      computeOuterProdCuda<0,false>(gaugeParam, coeff, thisField, neighborField, oprodField);
    }
  }
  return;
}




//*****************************************************************************************************************//
//*****************************************************************************************************************//
//*****************************************************************************************************************//
//*****************************************************************************************************************//

#else // single-gpu build

template<class Real, class RealA, bool forNaik> 
__global__ 
void computeOuterProd(const RealA* const colorVectorField, Real coeff, RealA* const oprodField)
{
  int sid = blockIdx.x*blockDim.x + threadIdx.x;

  int za, zb;
  int x1h, x2h;
  int x1, x2, x3, x4;
  int x1odd;
  int X;

  za  = sid/X1h;
  x1h = sid - za*X1h;
  zb  = za / X2;
  x2 = za - zb*X2;
  x4 = zb / X3;
  x3 = zb - x4*X3;
  x1odd = (x2 + x3 + x4) & 1;
  x1 = 2*x1h + x1odd;
  X = 2*sid + x1odd;

  RealA this_vector[3];     // colorVector at sid
  RealA neighbor_vector[3]; // colorVector at neighboring site
  RealA oprod[9];           // outer product

  int neighbor_index;


  // read the color vector at this lattice site
  // we only run on even half lattice sites
  readColorVectorFromField(colorVectorField, sid, sp_stride, this_vector);

  // hop in positive x direction
  // compute neighbor index
  if(forNaik){
    neighbor_index = ((x1 >= X1m3) ? X -X1m3 : X+3) >> 1;
  }else{
    neighbor_index = ((x1==X1m1) ? X-X1m1 : X+1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(neighbor_vector, this_vector, oprod);
  addOprodMatrixToField(oprod, 0, sid, coeff, oprodField); // offset + change of sign  


  // hop in the negative x direction
  // compute neighbor_index
  if(forNaik){
    neighbor_index = ((x1<3) ? X + X1m3: X -3)>>1; 
  }else{
    neighbor_index = ((x1==0) ? X+X1m1 : X-1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(this_vector, neighbor_vector, oprod);
  addOprodMatrixToField(oprod, 0, neighbor_index, coeff, oprodField+4*Vh*9);

  // hop in positive y direction
  // compute neighbor index
  if(forNaik){
    neighbor_index = ((x2 >= X2m3 ) ? X-X2m3*X1 : X+3*X1) >> 1;   
  }else{
    neighbor_index = ((x2==X2m1) ? X-X2X1mX1 : X+X1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(neighbor_vector, this_vector, oprod);
  addOprodMatrixToField(oprod, 1, sid, coeff, oprodField); // offset + change of sign  

  // hop in negative y direction
  // compute neighbor index
  if(forNaik){
    neighbor_index = ((x2 < 3) ? X + X2m3*X1: X-3*X1 )>> 1; 
  }else{
    neighbor_index = ((x2==0)    ? X+X2X1mX1 : X-X1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(this_vector, neighbor_vector, oprod);
  addOprodMatrixToField(oprod, 1, neighbor_index, coeff, oprodField+4*Vh*9);

  // hop in positive z direction
  // compute neighbor index
  if(forNaik){
    neighbor_index = ((x3>= X3m3)? X -X3m3*X2X1: X + 3*X2X1)>> 1;   
  }else{
    neighbor_index = ((x3==X3m1) ? X-X3X2X1mX2X1 : X+X2X1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(neighbor_vector, this_vector, oprod);
  addOprodMatrixToField(oprod, 2, sid, coeff, oprodField); // offset + change of sign  

  // hop in negative z direction
  // compute neighbor index
  if(forNaik){
    neighbor_index = ((x3 <3) ? X + X3m3*X2X1: X - 3*X2X1)>>1;
  }else{
    neighbor_index = ((x3==0)    ? X+X3X2X1mX2X1 : X-X2X1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(this_vector, neighbor_vector, oprod);
  addOprodMatrixToField(oprod, 2, neighbor_index, coeff, oprodField+4*Vh*9);

  // hop in positive t direction
  if(forNaik){
    neighbor_index = ((x4>=X4m3)? X -X4m3*X3X2X1 : X + 3*X3X2X1)>> 1;  
  }else{
    neighbor_index = ((x4==X4m1) ? X-X4X3X2X1mX3X2X1 : X+X3X2X1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(neighbor_vector, this_vector, oprod);
  addOprodMatrixToField(oprod, 3, sid, coeff, oprodField); // offset + change of sign  

  // hop in negative t direction 
  if(forNaik){
    neighbor_index = ((x4<3) ? X + X4m3*X3X2X1: X - 3*X3X2X1) >> 1;
  }else{
    neighbor_index = ((x4==0)  ? X+X4X3X2X1mX3X2X1 : X-X3X2X1) >> 1;
  }
  readColorVectorFromField(colorVectorField + 3*sp_stride, neighbor_index, sp_stride, neighbor_vector);
  computeOuterProduct(this_vector, neighbor_vector, oprod);
  addOprodMatrixToField(oprod, 3, neighbor_index, coeff,   oprodField+4*Vh*9);

  return;
}


void computeOuterProdCuda(const QudaGaugeParam& param,
			  double coeff,
			  const cudaColorSpinorField& quarkField,
			   cudaGaugeField* const oprodField)
{
  const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
  const int half_volume = volume/2;
  dim3 gridDim(half_volume/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM,1,1);

  if(param.cuda_prec == QUDA_SINGLE_PRECISION){
    cudaBindTexture(0, colorVecTexSingle, quarkField.V(), volume*24*sizeof(float));
    computeOuterProd<float,float2,false><<<gridDim,blockDim>>>((const float2* const)quarkField.V(), (float)coeff, (float2* const)oprodField->Gauge_p());
    cudaUnbindTexture(colorVecTexSingle);
  }else if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
    cudaBindTexture(0, colorVecTexDouble, quarkField.V(), volume*24*sizeof(double));
    computeOuterProd<double,double2,false><<<gridDim,blockDim>>>((const double2* const)quarkField.V(), coeff, (double2* const)oprodField->Gauge_p());
    cudaUnbindTexture(colorVecTexDouble);
  }else{
    errorQuda("Unrecognised precision\n");
    exit(1);
  }

  return;
}


void computeLongLinkOuterProdCuda(const QudaGaugeParam& param,
			          double coeff,
			          const cudaColorSpinorField& quarkField,
			          cudaGaugeField* const oprodField)
{
  const int volume = param.X[0]*param.X[1]*param.X[2]*param.X[3];
  const int half_volume = volume/2;
  dim3 gridDim(half_volume/BLOCK_DIM,1,1);
  dim3 blockDim(BLOCK_DIM,1,1);

  if(param.cuda_prec == QUDA_SINGLE_PRECISION){
    cudaBindTexture(0, colorVecTexSingle, quarkField.V(), volume*24*sizeof(float));
    computeOuterProd<float, float2, true><<<gridDim,blockDim>>>((const float2* const)quarkField.V(), (float)coeff, (float2* const)oprodField->Gauge_p());
    cudaUnbindTexture(colorVecTexSingle);
  }else if(param.cuda_prec == QUDA_DOUBLE_PRECISION){
    cudaBindTexture(0, colorVecTexDouble, quarkField.V(), volume*24*sizeof(double));
    computeOuterProd<double, double2, true><<<gridDim,blockDim>>>((const double2* const)quarkField.V(), coeff, (double2* const)oprodField->Gauge_p());
    cudaUnbindTexture(colorVecTexDouble);
  }else{
    errorQuda("Unrecognised precision\n");
    exit(1);
  }
  return;
}

#endif // single-gpu build

} // namespace fermion_force
} // namespace quda
