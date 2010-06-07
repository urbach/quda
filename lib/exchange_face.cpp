
#include <stdio.h>
#include <string.h>

#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include "exchange_face.h"
#include "mpicomm.h"
#include <sys/time.h>

static int V;
static int Vh;
static int Vs;
static int Vsh;
void* fwd_nbr_spinor_sendbuf = NULL;
void* back_nbr_spinor_sendbuf = NULL;
void* f_norm_sendbuf = NULL;
void* b_norm_sendbuf = NULL;
#define gaugeSiteSize 18
#define mySpinorSiteSize 6

void exchange_init(cudaColorSpinorField* cudaSpinor)
{
  static int exchange_initialized = 0;
  if (exchange_initialized){
    return;
  }
  exchange_initialized = 1;
  
  int len = 3*Vsh*mySpinorSiteSize*cudaSpinor->Precision();
  int normlen = 3*Vsh*sizeof(float);
  
  cudaMallocHost((void**)&fwd_nbr_spinor_sendbuf, len); CUERR;
  cudaMallocHost((void**)&back_nbr_spinor_sendbuf, len); CUERR;
  if (fwd_nbr_spinor_sendbuf == NULL || back_nbr_spinor_sendbuf == NULL){
    printf("ERROR: malloc failed for fwd_nbr_spinor_sendbuf/back_nbr_spinor_sendbuf\n"); 
    comm_exit(1);
  }
  
  if (cudaSpinor->Precision() == QUDA_HALF_PRECISION){
    cudaMallocHost(&f_norm_sendbuf, normlen);CUERR;
    cudaMallocHost(&b_norm_sendbuf, normlen);CUERR;
    if (f_norm_sendbuf == NULL || b_norm_sendbuf == NULL){
      printf("ERROR: malloc failed for b_norm_sendbuf/f_norm_sendbuf\n");
      comm_exit(1);
    }    
  }
  
  return;
}

void exchange_cleanup()
{    
  cudaFreeHost(fwd_nbr_spinor_sendbuf);
  cudaFreeHost(back_nbr_spinor_sendbuf);

  if (f_norm_sendbuf){
    cudaFreeHost(f_norm_sendbuf);
  }
  if (b_norm_sendbuf){
    cudaFreeHost(b_norm_sendbuf);
  }
}

template<typename Float>
void
exchange_fatlink(Float** fatlink, Float* ghost_fatlink, Float* fatlink_sendbuf)
{
  Float* even_fatlink_src = fatlink[3] + (Vh - Vsh)*gaugeSiteSize;
  Float* odd_fatlink_src = fatlink[3] + (V -Vsh)*gaugeSiteSize;
  
  Float* even_fatlink_dst = fatlink_sendbuf;
  Float* odd_fatlink_dst = fatlink_sendbuf + Vsh*gaugeSiteSize;

  int len = Vsh*gaugeSiteSize*sizeof(Float);
  memcpy(even_fatlink_dst, even_fatlink_src, len); 
  memcpy(odd_fatlink_dst, odd_fatlink_src, len);
  
  unsigned long recv_request = comm_recv(ghost_fatlink, 2*len, BACK_NBR);
  unsigned long send_request = comm_send(fatlink_sendbuf, 2*len, FWD_NBR);
  comm_wait(recv_request);
  comm_wait(send_request);
}

template<typename Float>
void
exchange_longlink(Float** longlink, Float* ghost_longlink, Float* longlink_sendbuf)
{
  Float* even_longlink_src = longlink[3] + (Vh -3*Vsh)*gaugeSiteSize;
  Float* odd_longlink_src = longlink[3] + (V - 3*Vsh)*gaugeSiteSize;
  
  Float* even_longlink_dst = longlink_sendbuf;
  Float* odd_longlink_dst = longlink_sendbuf + 3*Vsh*gaugeSiteSize;
  int len  = 3*Vsh*gaugeSiteSize*sizeof(Float);
  memcpy(even_longlink_dst, even_longlink_src, len);
  memcpy(odd_longlink_dst, odd_longlink_src, len);
  
  unsigned long recv_request = comm_recv(ghost_longlink, 2*len, BACK_NBR);
  unsigned long send_request = comm_send(longlink_sendbuf, 2*len, FWD_NBR);
  comm_wait(recv_request);
  comm_wait(send_request);
  
}

template<typename Float>
void
exchange_cpu_spinor(Float* spinorField, Float* fwd_nbr_spinor, Float* back_nbr_spinor)
{
  Float* fwd_nbr_spinor_send = spinorField + (Vh -3*Vsh)*mySpinorSiteSize;
  Float* back_nbr_spinor_send = spinorField;
  int len = 3*Vsh*mySpinorSiteSize*sizeof(Float);

  unsigned long recv_request1 = comm_recv(back_nbr_spinor, len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(fwd_nbr_spinor, len, FWD_NBR);
  
  unsigned long send_request1= comm_send(fwd_nbr_spinor_send, len, FWD_NBR);
  unsigned long send_request2 = comm_send(back_nbr_spinor_send, len, BACK_NBR);
  
  comm_wait(recv_request1);
  comm_wait(recv_request2);
  
  comm_wait(send_request1);
  comm_wait(send_request2);
}

void
exchange_gpu_spinor(cudaColorSpinorField* cudaSpinor, void* fwd_nbr_spinor, void* back_nbr_spinor, 
		    void* f_norm, void* b_norm, cudaStream_t* mystream)
{
 
  exchange_init(cudaSpinor);
  struct timeval t0, t1, t2, t3;
  
  int len = 3*Vsh*mySpinorSiteSize*cudaSpinor->Precision();
  int normlen = 3*Vsh*sizeof(float);

  gettimeofday(&t0, NULL);
  cudaSpinor->packGhostSpinor(fwd_nbr_spinor_sendbuf, back_nbr_spinor_sendbuf, f_norm_sendbuf, b_norm_sendbuf, mystream);
  cudaStreamSynchronize(*mystream);
  gettimeofday(&t1, NULL);
  
  unsigned long recv_request1 = comm_recv(back_nbr_spinor, len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(fwd_nbr_spinor, len, FWD_NBR);
  
  unsigned long send_request1= comm_send(fwd_nbr_spinor_sendbuf, len, FWD_NBR);
  unsigned long send_request2 = comm_send(back_nbr_spinor_sendbuf, len, BACK_NBR);

  unsigned long recv_request3 = 0;
  unsigned long recv_request4 = 0;
  unsigned long send_request3 = 0;
  unsigned long send_request4 = 0;
  
  if (cudaSpinor->Precision() == QUDA_HALF_PRECISION){
    recv_request3 = comm_recv(b_norm, normlen, BACK_NBR);
    recv_request4 = comm_recv(f_norm, normlen, FWD_NBR);
    send_request3 = comm_send(f_norm_sendbuf, normlen, FWD_NBR);
    send_request4 = comm_send(b_norm_sendbuf, normlen, BACK_NBR);
  }
  
  
  
  comm_wait(recv_request1);
  comm_wait(recv_request2);  
  comm_wait(send_request1);
  comm_wait(send_request2);
  
  if (cudaSpinor->Precision() == QUDA_HALF_PRECISION){
    comm_wait(recv_request3);
    comm_wait(recv_request4);
    comm_wait(send_request3);
    comm_wait(send_request4);
  }
  
  gettimeofday(&t2, NULL);
  cudaSpinor->unpackGhostSpinor(fwd_nbr_spinor, back_nbr_spinor, f_norm, b_norm, mystream);
  
  cudaStreamSynchronize(*mystream);
  gettimeofday(&t3, NULL);

  float pack_time = TDIFF(t1, t0)*1000;
  float send_recv_time = TDIFF(t2, t1)*1000;
  float unpack_time = TDIFF(t3, t2)*1000;
  
  PRINTF("Pack_time=%.2f(ms)(%.2f GB/s), send_recv_time=%.2f(ms)(%.2f GB/s), unpack_time=%.2f(ms)(%.2f GB/s)\n",
	 pack_time, 2*len/pack_time*1e-6, send_recv_time, 2*len/send_recv_time*1e-6, 
	 unpack_time, 2*len/unpack_time*1e-6);
  

}




void
exchange_gpu_spinor_start(cudaColorSpinorField* cudaSpinor, void* fwd_nbr_spinor, void* back_nbr_spinor, 
			  void* f_norm, void* b_norm, cudaStream_t* mystream)
{
 
  exchange_init(cudaSpinor);
  cudaSpinor->packGhostSpinor(fwd_nbr_spinor_sendbuf, back_nbr_spinor_sendbuf, f_norm_sendbuf, b_norm_sendbuf, mystream);
  
}

void
exchange_gpu_spinor_wait(cudaColorSpinorField* cudaSpinor, void* fwd_nbr_spinor, void* back_nbr_spinor, 
			 void* f_norm, void* b_norm, cudaStream_t* mystream)
{
 
  int len = 3*Vsh*mySpinorSiteSize*cudaSpinor->Precision();
  int normlen = 3*Vsh*sizeof(float);
  
  cudaStreamSynchronize(*mystream); //required the data to be there before sending out
  unsigned long recv_request1 = comm_recv(back_nbr_spinor, len, BACK_NBR);
  unsigned long recv_request2 = comm_recv(fwd_nbr_spinor, len, FWD_NBR);
  
  unsigned long send_request1= comm_send(fwd_nbr_spinor_sendbuf, len, FWD_NBR);
  unsigned long send_request2 = comm_send(back_nbr_spinor_sendbuf, len, BACK_NBR);

  unsigned long recv_request3 = 0;
  unsigned long recv_request4 = 0;
  unsigned long send_request3 = 0;
  unsigned long send_request4 = 0;
  
  if (cudaSpinor->Precision() == QUDA_HALF_PRECISION){
    recv_request3 = comm_recv(b_norm, normlen, BACK_NBR);
    recv_request4 = comm_recv(f_norm, normlen, FWD_NBR);
    send_request3 = comm_send(f_norm_sendbuf, normlen, FWD_NBR);
    send_request4 = comm_send(b_norm_sendbuf, normlen, BACK_NBR);
  }
  
  
  
  comm_wait(recv_request1);
  comm_wait(recv_request2);  
  comm_wait(send_request1);
  comm_wait(send_request2);
  
  if (cudaSpinor->Precision() == QUDA_HALF_PRECISION){
    comm_wait(recv_request3);
    comm_wait(recv_request4);
    comm_wait(send_request3);
    comm_wait(send_request4);
  }
  
  cudaSpinor->unpackGhostSpinor(fwd_nbr_spinor, back_nbr_spinor, f_norm, b_norm, mystream);  
  cudaStreamSynchronize(*mystream);
  
}




void exchange_cpu_links(int* X,
			void** fatlink, void* ghost_fatlink, 
			void** longlink, void* ghost_longlink,
			QudaPrecision gPrecision)
{
  
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
  }
  Vh = V/2;
  
  Vs = X[0]*X[1]*X[2];
  Vsh = Vs/2;
  
  void*  fatlink_sendbuf = malloc(Vs*gaugeSiteSize*gPrecision);
  void*  longlink_sendbuf = malloc(3*Vs*gaugeSiteSize*gPrecision);
  if (fatlink_sendbuf == NULL || longlink_sendbuf == NULL){
    printf("ERROR: malloc failed for fatlink_sendbuf/longlink_sendbuf\n");
    exit(1);
  }
  
 
  if (gPrecision == QUDA_DOUBLE_PRECISION){
    exchange_fatlink((double**)fatlink, (double*)ghost_fatlink, (double*)fatlink_sendbuf);
    exchange_longlink((double**)longlink, (double*)ghost_longlink, (double*)longlink_sendbuf);
  }else{ //single
    exchange_fatlink((float**)fatlink, (float*)ghost_fatlink, (float*)fatlink_sendbuf);
    exchange_longlink((float**)longlink, (float*)ghost_longlink, (float*)longlink_sendbuf);    
  }
  
  free(fatlink_sendbuf);
  free(longlink_sendbuf);

}


void exchange_cpu_spinor(int* X,
			 void* spinorField, void* fwd_nbr_spinor, void* back_nbr_spinor,
			 QudaPrecision sPrecision)
{
  
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
  }
  Vh = V/2;
  
  Vs = X[0]*X[1]*X[2];
  Vsh = Vs/2;

  if (sPrecision == QUDA_DOUBLE_PRECISION){
    exchange_cpu_spinor((double*)spinorField, (double*)fwd_nbr_spinor, (double*)back_nbr_spinor);
  }else{//single
    exchange_cpu_spinor((float*)spinorField, (float*)fwd_nbr_spinor, (float*)back_nbr_spinor);    
  }

}
