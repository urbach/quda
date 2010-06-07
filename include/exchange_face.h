#ifndef __EXCHANGE_FACE_H__
#define __EXCHANGE_FACE_H__
#include <color_spinor_field.h>

void exchange_cpu_links(int* X,
			void** fatlink, void* ghost_fatlink, 
			void** longlink, void* ghost_longlink,
			QudaPrecision gPrecision);
void exchange_cpu_spinor(int* X,
			 void* spinorField, void* fwd_nbr_spinor, void* back_nbr_spinor,
			 QudaPrecision sPrecision);
void exchange_gpu_spinor(cudaColorSpinorField* cudaSpinor,
			 void* fwd_nbr_spinor, void* back_nbr_spinor,
			 void* f_norm, void* b_norm, cudaStream_t* stream);
void exchange_gpu_spinor_start(cudaColorSpinorField* cudaSpinor,
			       void* fwd_nbr_spinor, void* back_nbr_spinor,
			       void* f_norm, void* b_norm, cudaStream_t* stream);
void exchange_gpu_spinor_wait(cudaColorSpinorField* cudaSpinor,
			      void* fwd_nbr_spinor, void* back_nbr_spinor,
			      void* f_norm, void* b_norm, cudaStream_t* stream);

#define TDIFF(t1, t0) ((t1.tv_sec - t0.tv_sec) + 1e-6*(t1.tv_usec -t0.tv_usec))

#endif



