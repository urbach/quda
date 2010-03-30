#include <stdlib.h>
#include <stdio.h>

#include <quda.h>
#include <gauge_quda.h>

#include <xmmintrin.h>

#define SHORT_LENGTH 65536
#define SCALE_FLOAT (SHORT_LENGTH-1) / 2.f
#define SHIFT_FLOAT -1.f / (SHORT_LENGTH-1)

double Anisotropy;

template <typename Float>
inline short FloatToShort(Float a) {
  return (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
}

template <typename Float>
inline void pack8(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*4*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  r[V].x = g[2];
  r[V].y = g[3];
  r[2*V].x = g[4];
  r[2*V].y = g[5];
  r[3*V].x = g[6];
  r[3*V].y = g[7];
}

template <typename Float>
inline void pack8(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*2*V;
  r[0].x = atan2(g[1], g[0]);
  r[0].y = atan2(g[13], g[12]);
  r[0].z = g[2];
  r[0].w = g[3];
  r[V].x = g[4];
  r[V].y = g[5];
  r[V].z = g[6];
  r[V].w = g[7];
}

template <typename Float>
inline void pack8(short4 *res, Float *g, int dir, int V) {
    short4 *r = res + dir*2*V;
    r[0].x = FloatToShort(atan2(g[1], g[0])/ M_PI);
    r[0].y = FloatToShort(atan2(g[13], g[12])/ M_PI);
    r[0].z = FloatToShort(g[2]);
    r[0].w = FloatToShort(g[3]);
    r[V].x = FloatToShort(g[4]);
    r[V].y = FloatToShort(g[5]);
    r[V].z = FloatToShort(g[6]);
    r[V].w = FloatToShort(g[7]);
       
}

template <typename Float>
inline void pack12(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*6*V;
  for (int j=0; j<6; j++) {
    r[j*V].x = g[j*2+0]; 
    r[j*V].y = g[j*2+1]; 
  }
}

template <typename Float>
inline void pack12(float4 *res, Float *g, int dir, int V) {
  float4 *r = res + dir*3*V;
  for (int j=0; j<3; j++) {
    r[j*V].x = (float)g[j*4+0]; 
    r[j*V].y = (float)g[j*4+1]; 
    r[j*V].z = (float)g[j*4+2]; 
    r[j*V].w = (float)g[j*4+3];
  }
}

template <typename Float>
inline void pack12(short4 *res, Float *g, int dir, int V) {
  short4 *r = res + dir*3*V;
  for (int j=0; j<3; j++) {
    r[j*V].x = FloatToShort(g[j*4+0]);
    r[j*V].y = FloatToShort(g[j*4+1]);
    r[j*V].z = FloatToShort(g[j*4+2]);
    r[j*V].w = FloatToShort(g[j*4+3]);
  }
}

template <typename Float>
inline void pack18(double2 *res, Float *g, int dir, int V) {
  double2 *r = res + dir*9*V;
  for (int j=0; j<9; j++) {
    r[j*V].x = g[j*2+0]; 
    r[j*V].y = g[j*2+1]; 
  }
}

template <typename Float>
inline void pack18(float2 *res, Float *g, int dir, int V) 
{
    float2 *r = res + dir*9*V;
    for (int j=0; j<9; j++) {
	r[j*V].x = (float)g[j*2+0]; 
	r[j*V].y = (float)g[j*2+1]; 
    }
}

template <typename Float>
inline void pack18(short2 *res, Float *g, int dir, int V) 
{
    short2 *r = res + dir*9*V;
    for (int j=0; j<9; j++) {
	r[j*V].x = FloatToShort(g[j*2+0]); 
	r[j*V].y = FloatToShort(g[j*2+1]); 
    }
}


template <typename Float>
inline void pack12(float2 *res, Float *g, int dir, int V){printf("ERROR: %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void pack12(short2 *res, Float *g, int dir, int V){printf("ERROR: %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void pack8(float2 *res, Float *g, int dir, int V){printf("ERROR: %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void pack8(short2 *res, Float *g, int dir, int V){printf("ERROR: %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void pack18(float4 *res, Float *g, int dir, int ){printf("ERROR: 0 %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void pack18(short4 *res, Float *g, int dir, int V){printf("ERROR:1 %s is called\n", __FUNCTION__); exit(1);}

// Assume the gauge field is "QDP" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
void packQDPGaugeField(FloatN *res, Float **gauge, int oddBit, ReconstructType reconstruct, int V) {
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge[dir] + oddBit*V*18;
      for (int i = 0; i < V; i++) pack12(res+i, g+i*18, dir, V);
    }
  } else if (reconstruct == QUDA_RECONSTRUCT_8){
      for (int dir = 0; dir < 4; dir++) {
	  Float *g = gauge[dir] + oddBit*V*18;
	  for (int i = 0; i < V; i++) pack8(res+i, g+i*18, dir, V);
      }
  }else{
      for (int dir = 0; dir < 4; dir++) {
	  Float *g = gauge[dir] + oddBit*V*18;
	  for (int i = 0; i < V; i++) {
	      pack18(res+i, g+i*18, dir, V);
	  }
      }      
  }
}

// Assume the gauge field is "Wilson" ordered directions inside of
// space-time column-row ordering even-odd space-time
template <typename Float, typename FloatN>
void packCPSGaugeField(FloatN *res, Float *gauge, int oddBit, ReconstructType reconstruct, int V) {
  Float gT[18];
  if (reconstruct == QUDA_RECONSTRUCT_12) {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	// Must reorder rows-columns and divide by anisotropy
	for (int ic=0; ic<2; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r] / Anisotropy;
	pack12(res+i, gT, dir, V);
      }
    } 
  } else {
    for (int dir = 0; dir < 4; dir++) {
      Float *g = gauge + (oddBit*V*4+dir)*18;
      for (int i = 0; i < V; i++) {
	// Must reorder rows-columns and divide by anisotropy
	for (int ic=0; ic<3; ic++) for (int jc=0; jc<3; jc++) for (int r=0; r<2; r++)
	  gT[(ic*3+jc)*2+r] = g[4*i*18 + (jc*3+ic)*2+r] / Anisotropy;
	pack8(res+i, gT, dir, V);
      }
    }
  }

}

void allocateGaugeField(FullGauge *cudaGauge, ReconstructType reconstruct, Precision precision) {

  cudaGauge->reconstruct = reconstruct;
  cudaGauge->precision = precision;

  cudaGauge->Nc = 3;

  int floatSize;
  if (precision == QUDA_DOUBLE_PRECISION) floatSize = sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) floatSize = sizeof(float);
  else floatSize = sizeof(float)/2;

  int elements;
  switch(reconstruct){
  case QUDA_RECONSTRUCT_8:
      elements = 8;
      break;
  case QUDA_RECONSTRUCT_12:
      elements = 12;
      break;
  case QUDA_RECONSTRUCT_NO:
      elements = 18;
      break;
  default:
      fprintf(stderr, "ERROR: no such reconstruct type(%d)\n", reconstruct);
      exit(1);
  }


  cudaGauge->bytes = 4*cudaGauge->volume*elements*floatSize;
  
  if (!cudaGauge->even) {
      if (cudaMalloc((void **)&cudaGauge->even, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
	  printf("Error allocating even gauge field\n");
	  exit(0);
      }
  }

  cudaMemset(cudaGauge->even, 0, cudaGauge->bytes); CUERR;


  if (!cudaGauge->odd) {
      if (cudaMalloc((void **)&cudaGauge->odd, cudaGauge->bytes) == cudaErrorMemoryAllocation) {
	  printf("Error allocating even odd gauge field\n");
	  exit(0);
      }
  }

  cudaMemset(cudaGauge->odd, 0, cudaGauge->bytes); CUERR;  

}


/******************************** Functions to manipulate Staple ****************************/

static void 
allocateStapleQuda(FullStaple *cudaStaple, Precision precision) 
{
    cudaStaple->precision = precision;    
    cudaStaple->Nc = 3;
    
    int floatSize;
    if (precision == QUDA_DOUBLE_PRECISION) {
	floatSize = sizeof(double);
    }
    else if (precision == QUDA_SINGLE_PRECISION) {
	floatSize = sizeof(float);
    }else{
	printf("ERROR: stape does not support half precision\n");
	exit(1);
    }
    
    int elements = 18;
    
    cudaStaple->bytes = cudaStaple->volume*elements*floatSize;
    
    cudaMalloc((void **)&cudaStaple->even, cudaStaple->bytes);CUERR;
    cudaMalloc((void **)&cudaStaple->odd, cudaStaple->bytes); CUERR;	    
}

void
createStapleQuda(FullStaple* cudaStaple, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
	printf("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
	exit(-1);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
	exit(-1);
    }
    
    cudaStaple->volume = 1;
    for (int d=0; d<4; d++) {
	cudaStaple->X[d] = param->X[d];
	cudaStaple->volume *= param->X[d];
    }
    cudaStaple->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaStaple->volume /= 2;    
    cudaStaple->blockDim = param->blockDim;
    
    allocateStapleQuda(cudaStaple,  param->cuda_prec);
    
    return;
}


void
freeStapleQuda(FullStaple *cudaStaple) 
{
    if (cudaStaple->even) {
	cudaFree(cudaStaple->even);
    }
    if (cudaStaple->odd) {
	cudaFree(cudaStaple->odd);
    }
    cudaStaple->even = NULL;
    cudaStaple->odd = NULL;
}



/******************************** Functions to manipulate Mom ****************************/

static void 
allocateMomQuda(FullMom *cudaMom, Precision precision) 
{
    cudaMom->precision = precision;    
    
    int floatSize;
    if (precision == QUDA_DOUBLE_PRECISION) {
	floatSize = sizeof(double);
    }
    else if (precision == QUDA_SINGLE_PRECISION) {
	floatSize = sizeof(float);
    }else{
	printf("ERROR: stape does not support half precision\n");
	exit(1);
    }
    
    int elements = 10;
     
    cudaMom->bytes = cudaMom->volume*elements*floatSize*4;
    
    cudaMalloc((void **)&cudaMom->even, cudaMom->bytes);CUERR;
    cudaMalloc((void **)&cudaMom->odd, cudaMom->bytes); CUERR;	    
}

void
createMomQuda(FullMom* cudaMom, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
	printf("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
	exit(-1);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
	exit(-1);
    }
    
    cudaMom->volume = 1;
    for (int d=0; d<4; d++) {
	cudaMom->X[d] = param->X[d];
	cudaMom->volume *= param->X[d];
    }
    cudaMom->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaMom->volume /= 2;    
    cudaMom->blockDim = param->blockDim;
    
    allocateMomQuda(cudaMom,  param->cuda_prec);
    
    return;
}


void
freeMomQuda(FullMom *cudaMom) 
{
    if (cudaMom->even) {
	cudaFree(cudaMom->even);
    }
    if (cudaMom->odd) {
	cudaFree(cudaMom->odd);
    }
    cudaMom->even = NULL;
    cudaMom->odd = NULL;
}

template <typename Float, typename Float2>
inline void pack10(Float2 *res, Float *m, int dir, int Vh) 
{
    Float2 *r = res + dir*5*Vh;
    for (int j=0; j<5; j++) {
	r[j*Vh].x = (float)m[j*2+0]; 
	r[j*Vh].y = (float)m[j*2+1]; 
    }
}

template <typename Float, typename Float2>
void packMomField(Float2 *res, Float *mom, int oddBit, int Vh) 
{    
    for (int dir = 0; dir < 4; dir++) {
	Float *g = mom + (oddBit*Vh*4 + dir)*momSiteSize;
	for (int i = 0; i < Vh; i++) {
	    pack10(res+i, g + 4*i*momSiteSize, dir, Vh);
	}
    }      
}

template <typename Float, typename Float2>
void loadMomField(Float2 *even, Float2 *odd, Float *mom,
		  int bytes, int Vh) 
{
    
    Float2 *packedEven, *packedOdd;
    cudaMallocHost((void**)&packedEven, bytes); CUERR;
    cudaMallocHost((void**)&packedOdd, bytes);CUERR;
    
    packMomField(packedEven, (Float*)mom, 0, Vh);
    packMomField(packedOdd,  (Float*)mom, 1, Vh);
    
    cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);CUERR;
    cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice); CUERR;
    
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
}




void
loadMomToGPU(FullMom cudaMom, void* mom, QudaGaugeParam* param)
{
    if (param->cuda_prec == QUDA_DOUBLE_PRECISION) {
	//loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
	//cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    } else { //single precision
	loadMomField((float2*)(cudaMom.even), (float2*)(cudaMom.odd), (float*)mom, 
		     cudaMom.bytes, cudaMom.volume);
	
    }
}

template <typename Float, typename Float2>
inline void unpack10(Float* m, Float2 *res, int dir, int Vh) 
{
    Float2 *r = res + dir*5*Vh;
    for (int j=0; j<5; j++) {
	m[j*2+0] = r[j*Vh].x;
	m[j*2+1] = r[j*Vh].y;
    }
    
}

template <typename Float, typename Float2>
void 
unpackMomField(Float* mom, Float2 *res, int oddBit, int Vh) 
{
    int dir, i;
    Float *m = mom + oddBit*Vh*momSiteSize*4;
    
    for (i = 0; i < Vh; i++) {
	for (dir = 0; dir < 4; dir++) {	
	    Float* thismom = m + (4*i+dir)*momSiteSize;
	    unpack10(thismom, res+i, dir, Vh);
	}
    }
}

template <typename Float, typename Float2>
void 
storeMomToCPUArray(Float* mom, Float2 *even, Float2 *odd, 
		   int bytes, int Vh) 
{    
    Float2 *packedEven, *packedOdd;   
    cudaMallocHost((void**)&packedEven, bytes); CUERR;
    cudaMallocHost((void**)&packedOdd, bytes); CUERR;
    cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost); CUERR;
    cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);  CUERR;

    unpackMomField((Float*)mom, packedEven,0, Vh);
    unpackMomField((Float*)mom, packedOdd, 1, Vh);
        
    cudaFreeHost(packedEven); CUERR;
    cudaFreeHost(packedOdd); CUERR;
}

void 
storeMomToCPU(void* mom, FullMom cudaMom, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec != cuda_prec){
	printf("Error:%s: cpu and gpu precison has to be the same at this moment \n", __FUNCTION__);
	exit(1);	
    }
    
    if (cpu_prec == QUDA_HALF_PRECISION){
	printf("ERROR: %s:  half precision is not supported at this moment\n", __FUNCTION__);
	exit(1);
    }
    
    if (cpu_prec == QUDA_DOUBLE_PRECISION){
	
    }else { //SINGLE PRECISIONS
	storeMomToCPUArray( (float*)mom, (float2*) cudaMom.even, (float2*)cudaMom.odd, 
			    cudaMom.bytes, cudaMom.volume);	
    }
    
}

/**************************************************************************************************/


void freeGaugeField(FullGauge *cudaGauge) {
  if (cudaGauge->even) cudaFree(cudaGauge->even);
  if (cudaGauge->odd) cudaFree(cudaGauge->odd);
  cudaGauge->even = NULL;
  cudaGauge->odd = NULL;
}



template <typename Float, typename FloatN>
void loadGaugeField(FloatN *even, FloatN *odd, Float *cpuGauge, ReconstructType reconstruct, 
		    int bytes, int Vh) {

  // Use pinned memory
  
  FloatN *packedEven, *packedOdd;
    

#ifndef __DEVICE_EMULATION__
  cudaMallocHost((void**)&packedEven, bytes);
  cudaMallocHost((void**)&packedOdd, bytes);
#else
  packedEven = (FloatN*)malloc(bytes);
  packedOdd = (FloatN*)malloc(bytes);
#endif
    
  if (gauge_param->gauge_order == QUDA_QDP_GAUGE_ORDER) {
      packQDPGaugeField(packedEven, (Float**)cpuGauge, 0, reconstruct, Vh);
      packQDPGaugeField(packedOdd,  (Float**)cpuGauge, 1, reconstruct, Vh);
  } else if (gauge_param->gauge_order == QUDA_CPS_WILSON_GAUGE_ORDER) {
      packCPSGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh);
      packCPSGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh);
    
  } else {
    printf("Sorry, %d GaugeFieldOrder not supported\n", gauge_param->gauge_order);
    exit(-1);
  }
  
  cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);    
    
#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
#else
  free(packedEven);
  free(packedOdd);
#endif

}


void 
createGaugeField(FullGauge *cudaGauge, void *cpuGauge, ReconstructType reconstruct, 
		 Precision precision, int *X, double anisotropy, int blockDim) 
{
    
    if (gauge_param->cpu_prec == QUDA_HALF_PRECISION) {
	printf("QUDA error: half precision not supported on cpu\n");
	exit(-1);
    }

    if (precision == QUDA_DOUBLE_PRECISION && gauge_param->cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
	exit(-1);
    }

    Anisotropy = anisotropy;

    cudaGauge->anisotropy = anisotropy;
    cudaGauge->volume = 1;
    for (int d=0; d<4; d++) {
	cudaGauge->X[d] = X[d];
	cudaGauge->volume *= X[d];
    }
    cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaGauge->volume /= 2;

    cudaGauge->blockDim = blockDim;
    
    allocateGaugeField(cudaGauge, reconstruct, precision);

    if (precision == QUDA_DOUBLE_PRECISION) {
	loadGaugeField((double2*)(cudaGauge->even), (double2*)(cudaGauge->odd), (double*)cpuGauge, 
		       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
    } else if (precision == QUDA_SINGLE_PRECISION) {
	if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION){
	    if (reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (double*)cpuGauge, 
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	    }else{
		loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (double*)cpuGauge, 
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);		
	    }
	}
	else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION){
	    if (reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeField((float4*)(cudaGauge->even), (float4*)(cudaGauge->odd), (float*)cpuGauge, 
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	    }else{
		loadGaugeField((float2*)(cudaGauge->even), (float2*)(cudaGauge->odd), (float*)cpuGauge, 
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	    }
	}
    } else if (precision == QUDA_HALF_PRECISION) {
	if (gauge_param->cpu_prec == QUDA_DOUBLE_PRECISION){
	    if (reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (double*)cpuGauge, 
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	    }else{
		loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (double*)cpuGauge, 
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	    }
	}
	else if (gauge_param->cpu_prec == QUDA_SINGLE_PRECISION){
	    if (reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeField((short4*)(cudaGauge->even), (short4*)(cudaGauge->odd), (float*)cpuGauge,
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	    }else{
		loadGaugeField((short2*)(cudaGauge->even), (short2*)(cudaGauge->odd), (float*)cpuGauge,
			       cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);		
	    }
	}
    }
}

template <typename Float, typename FloatN>
void 
packGaugeField(FloatN *res, Float *gauge, int oddBit, ReconstructType reconstruct, int Vh) 
{
    int dir, i;
    if (reconstruct == QUDA_RECONSTRUCT_12) {
	for (dir = 0; dir < 4; dir++) {
	    Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
	    for (i = 0; i < Vh; i++) {
		pack12(res+i, g+4*i*gaugeSiteSize+dir*gaugeSiteSize, dir, Vh);
	    }
	}
    } else if (reconstruct == QUDA_RECONSTRUCT_8){
	for (dir = 0; dir < 4; dir++) {
	    Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
	    for (i = 0; i < Vh; i++) {
		pack8(res+i, g+4*i*gaugeSiteSize + dir*gaugeSiteSize, dir, Vh);
	    }
	}
    }else{
	for (dir = 0; dir < 4; dir++) {
	    Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
	    for (i = 0; i < Vh; i++) {
		pack18(res+i, g+i*gaugeSiteSize+dir, dir*gaugeSiteSize, Vh);
	    }
	}      
    }
}

template <typename Float, typename FloatN>
void 
loadGaugeFromCPUArrayQuda(FloatN *even, FloatN *odd, Float *cpuGauge, 
			  ReconstructType reconstruct, int bytes, int Vh) 
{
    
    // Use pinned memory
    
    FloatN *packedEven, *packedOdd;    
    cudaMallocHost((void**)&packedEven, bytes);
    cudaMallocHost((void**)&packedOdd, bytes);
    
    
    packGaugeField(packedEven, (Float*)cpuGauge, 0, reconstruct, Vh);
    packGaugeField(packedOdd,  (Float*)cpuGauge, 1, reconstruct, Vh);
    
    
    cudaMemcpy(even, packedEven, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(odd,  packedOdd, bytes, cudaMemcpyHostToDevice);    
    
    cudaFreeHost(packedEven);
    cudaFreeHost(packedOdd);
}




void
createLinkQuda(FullGauge* cudaGauge, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec == QUDA_HALF_PRECISION) {
	printf("ERROR: %s:  half precision not supported on cpu\n", __FUNCTION__);
	exit(-1);
    }
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION && param->cpu_prec != QUDA_DOUBLE_PRECISION) {
	printf("Error: can only create a double GPU gauge field from a double CPU gauge field\n");
	exit(-1);
    }
        
    cudaGauge->anisotropy = param->anisotropy;
    cudaGauge->volume = 1;
    for (int d=0; d<4; d++) {
	cudaGauge->X[d] = param->X[d];
	cudaGauge->volume *= param->X[d];
    }
    cudaGauge->X[0] /= 2; // actually store the even-odd sublattice dimensions
    cudaGauge->volume /= 2;    
    cudaGauge->blockDim = param->blockDim;
    cudaGauge->reconstruct = param->reconstruct;

    allocateGaugeField(cudaGauge, param->reconstruct, param->cuda_prec);
    
    return;
}

void 
loadLinkToGPU(FullGauge cudaGauge, void *cpuGauge, QudaGaugeParam* param)
{
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cuda_prec == QUDA_DOUBLE_PRECISION) {
	loadGaugeFromCPUArrayQuda((double2*)(cudaGauge.even), (double2*)(cudaGauge.odd), (double*)cpuGauge, 
				  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
    } else if (cuda_prec == QUDA_SINGLE_PRECISION) {
	if (cpu_prec == QUDA_DOUBLE_PRECISION){
	    if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeFromCPUArrayQuda((float4*)(cudaGauge.even), (float4*)(cudaGauge.odd), (double*)cpuGauge, 
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
	    }else{
		loadGaugeFromCPUArrayQuda((float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (double*)cpuGauge, 
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);		
	    }
	}
	else if (cpu_prec == QUDA_SINGLE_PRECISION){
	    if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeFromCPUArrayQuda((float4*)(cudaGauge.even), (float4*)(cudaGauge.odd), (float*)cpuGauge, 
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
	    }else{
		loadGaugeFromCPUArrayQuda((float2*)(cudaGauge.even), (float2*)(cudaGauge.odd), (float*)cpuGauge, 
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
	    }
	}
    } else if (cuda_prec == QUDA_HALF_PRECISION) {
	if (cpu_prec == QUDA_DOUBLE_PRECISION){
	    if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeFromCPUArrayQuda((short4*)(cudaGauge.even), (short4*)(cudaGauge.odd), (double*)cpuGauge, 
			       cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
	    }else{
		loadGaugeFromCPUArrayQuda((short2*)(cudaGauge.even), (short2*)(cudaGauge.odd), (double*)cpuGauge, 
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
	    }
	}
	else if (cpu_prec == QUDA_SINGLE_PRECISION){
	    if (cudaGauge.reconstruct != QUDA_RECONSTRUCT_NO){
		loadGaugeFromCPUArrayQuda((short4*)(cudaGauge.even), (short4*)(cudaGauge.odd), (float*)cpuGauge,
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);
	    }else{
		loadGaugeFromCPUArrayQuda((short2*)(cudaGauge.even), (short2*)(cudaGauge.odd), (float*)cpuGauge,
					  cudaGauge.reconstruct, cudaGauge.bytes, cudaGauge.volume);		
	    }
	}
    }
}
/*****************************************************************/
/********************** store link data to cpu memory ************/
template <typename Float>
inline void unpack12(Float* g, float2 *res, int dir, int V){printf("ERROR: %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void unpack8(Float* g, float2 *res,  int dir, int V){printf("ERROR: %s is called\n", __FUNCTION__); exit(1);}
template <typename Float>
inline void unpack18(Float* g, float4 *res, int dir, int ){printf("ERROR: 0 %s is called\n", __FUNCTION__); exit(1);}

template <typename Float>
inline void unpack18(Float* g, double2 *res, int dir, int V) 
{
    double2 *r = res + dir*9*V;
    for (int j=0; j<9; j++) {
	g[j*2+0] = r[j*V].x;
	g[j*2+1] = r[j*V].y;
    }
}

template <typename Float>
inline void unpack18(Float* g, float2 *res, int dir, int V) 
{
    float2 *r = res + dir*9*V;
    for (int j=0; j<9; j++) {
	g[j*2+0] = r[j*V].x;
	g[j*2+1] = r[j*V].y;
    }

}
template <typename Float, typename FloatN>
void 
unpackGaugeField(Float* gauge, FloatN *res, int oddBit, ReconstructType reconstruct, int Vh) 
{
    int dir, i;
    if (reconstruct == QUDA_RECONSTRUCT_12) {
	for (dir = 0; dir < 4; dir++) {
	    //Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
	    for (i = 0; i < Vh; i++) {
		//unpack12(g+4*i*gaugeSiteSize+dir*gaugeSiteSize, res+i, dir, Vh);
	    }
	}
    } else if (reconstruct == QUDA_RECONSTRUCT_8){
	for (dir = 0; dir < 4; dir++) {
	    //Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
	    for (i = 0; i < Vh; i++) {
		//unpack8(g+4*i*gaugeSiteSize + dir*gaugeSiteSize, res+i, dir, Vh);
	    }
	}
    }else{
	for (dir = 0; dir < 4; dir++) {
	    Float *g = gauge + oddBit*Vh*gaugeSiteSize*4;
	    for (i = 0; i < Vh; i++) {
		unpack18(g+4*i*gaugeSiteSize+dir*gaugeSiteSize, res+i,dir, Vh);
	    }
	}      
    }
}

template <typename Float, typename FloatN>
void 
storeGaugeToCPUArray(Float* cpuGauge, FloatN *even, FloatN *odd, 
		     ReconstructType reconstruct, int bytes, int Vh) 
{
    
    // Use pinned memory
    
    FloatN *packedEven, *packedOdd;    
    cudaMallocHost((void**)&packedEven, bytes); CUERR;
    cudaMallocHost((void**)&packedOdd, bytes); CUERR;
    cudaMemcpy(packedEven, even, bytes, cudaMemcpyDeviceToHost); CUERR;
    cudaMemcpy(packedOdd, odd, bytes, cudaMemcpyDeviceToHost);  CUERR;
    
    
    unpackGaugeField((Float*)cpuGauge, packedEven,0, reconstruct, Vh);
    unpackGaugeField((Float*)cpuGauge, packedOdd, 1, reconstruct, Vh);
    
    cudaFreeHost(packedEven); CUERR;
    cudaFreeHost(packedOdd); CUERR;
}


void 
storeLinkToCPU(void* cpuGauge, FullGauge *cudaGauge, QudaGaugeParam* param)
{
    
    QudaPrecision cpu_prec = param->cpu_prec;
    QudaPrecision cuda_prec= param->cuda_prec;
    
    if (cpu_prec != cuda_prec){
	printf("Error: cpu and gpu precison has to be the same at this moment\n");
	exit(1);	
    }
    
    if (cpu_prec == QUDA_HALF_PRECISION){
	printf("ERROR: %s:  half precision is not supported at this moment\n", __FUNCTION__);
	exit(1);
    }
    
    if (cpu_prec == QUDA_DOUBLE_PRECISION){
	
    }else { //SINGLE PRECISIONS
	if (cudaGauge->reconstruct == QUDA_RECONSTRUCT_NO){
	    storeGaugeToCPUArray( (float*)cpuGauge, (float2*) cudaGauge->even, (float2*)cudaGauge->odd, 
				  cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);
	}else{
	    storeGaugeToCPUArray( (float*)cpuGauge, (float4*) cudaGauge->even, (float4*)cudaGauge->odd, 
				  cudaGauge->reconstruct, cudaGauge->bytes, cudaGauge->volume);	    
	}
	
    }

}
