#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include <invert_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>

#include <blas_reference.h>
#include "misc.h"

extern int Vh;
extern int V;

FullGauge cudaGaugePrecise; // precise gauge field
FullGauge cudaGaugeSloppy; // sloppy gauge field

FullGauge cudaFatLinkPrecise; 
FullGauge cudaFatLinkSloppy;
FullGauge cudaLongLinkPrecise; 
FullGauge cudaLongLinkSloppy; 

void printGaugeParam(QudaGaugeParam *param) {

  printf("Gauge Params:\n");
  for (int d=0; d<4; d++) {
    printf("X[%d] = %d\n", d, param->X[d]);
  }
  printf("anisotropy = %e\n", param->anisotropy);
  printf("gauge_order = %d\n", param->gauge_order);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("reconstruct = %d\n", param->reconstruct);
  printf("cuda_prec_sloppy = %d\n", param->cuda_prec_sloppy);
  printf("reconstruct_sloppy = %d\n", param->reconstruct_sloppy);
  printf("gauge_fix = %d\n", param->gauge_fix);
  printf("t_boundary = %d\n", param->t_boundary);
  printf("packed_size = %d\n", param->packed_size);
  printf("gaugeGiB = %e\n", param->gaugeGiB);
}

void printInvertParam(QudaInvertParam *param) {
  printf("kappa = %e\n", param->kappa);
  printf("mass_normalization = %d\n", param->mass_normalization);
  printf("inv_type = %d\n", param->inv_type);
  printf("tol = %e\n", param->tol);
  printf("iter = %d\n", param->iter);
  printf("maxiter = %d\n", param->maxiter);
  printf("matpc_type = %d\n", param->matpc_type);
  printf("solution_type = %d\n", param->solution_type);
  printf("preserve_source = %d\n", param->preserve_source);
  printf("cpu_prec = %d\n", param->cpu_prec);
  printf("cuda_prec = %d\n", param->cuda_prec);
  printf("dirac_order = %d\n", param->dirac_order);
  printf("spinorGiB = %e\n", param->spinorGiB);
  printf("gflops = %e\n", param->gflops);
  printf("secs = %f\n", param->secs);
}

void initQuda(int dev)
{
    static int init_quda_flag = 0 ;
    if (init_quda_flag){
	return;
    }
    init_quda_flag = 1;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
	fprintf(stderr, "No devices supporting CUDA.\n");
	exit(EXIT_FAILURE);
    }
    
    for(int i=0; i<deviceCount; i++) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, i);
	PRINTF("found device %d: %s\n", i, deviceProp.name);
    }
    
    if(dev<0) {
	//dev = deviceCount - 1;
	dev = 0;
    }
    //dev = deviceCount-1;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (deviceProp.major < 1) {
	fprintf(stderr, "Device %d does not support CUDA.\n", dev);
	exit(EXIT_FAILURE);
    }
    
    PRINTF("Using device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);CUERR;
    
    cudaGaugePrecise.even = NULL;
    cudaGaugePrecise.odd = NULL;
    
    cudaGaugeSloppy.even = NULL;
    cudaGaugeSloppy.odd = NULL;
    
}
void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
    gauge_param = param;
    
    gauge_param->packed_size = (gauge_param->reconstruct == QUDA_RECONSTRUCT_8) ? 8 : 12;
    
    createGaugeField(&cudaGaugePrecise, h_gauge, gauge_param->reconstruct,
		     gauge_param->cuda_prec, gauge_param->X, gauge_param->anisotropy, gauge_param->blockDim);
    gauge_param->gaugeGiB = 2.0*cudaGaugePrecise.bytes/ (1 << 30);
    if (gauge_param->cuda_prec_sloppy != gauge_param->cuda_prec ||
	gauge_param->reconstruct_sloppy != gauge_param->reconstruct) {
	createGaugeField(&cudaGaugeSloppy, h_gauge, gauge_param->reconstruct_sloppy,
			 gauge_param->cuda_prec_sloppy, gauge_param->X, gauge_param->anisotropy,
			 gauge_param->blockDim_sloppy);
	gauge_param->gaugeGiB += 2.0*cudaGaugeSloppy.bytes/ (1 << 30);
    } else {
	cudaGaugeSloppy = cudaGaugePrecise;
    }    
}

void loadGaugeQuda_st(void *h_gauge, QudaGaugeParam *param, void* _cudaLinkPrecise, void* _cudaLinkSloppy)
{
    FullGauge* cudaLinkPrecise = (FullGauge*)_cudaLinkPrecise;
    FullGauge* cudaLinkSloppy = (FullGauge*)_cudaLinkSloppy;
    
    gauge_param = param;
    
    int packed_size;
    switch(gauge_param->reconstruct){
    case QUDA_RECONSTRUCT_8:
	packed_size = 8;
	break;
    case QUDA_RECONSTRUCT_12:
	packed_size = 12;
	break;
    case QUDA_RECONSTRUCT_NO:
	packed_size = 18;
	break;
    default:
	printf("ERROR: %s: reconstruct type not set, exitting\n", __FUNCTION__);
	exit(1);
    }
    
    gauge_param->packed_size = packed_size;
   
    createGaugeField(cudaLinkPrecise, h_gauge, gauge_param->reconstruct, 
		     gauge_param->cuda_prec, gauge_param->X, gauge_param->anisotropy, gauge_param->blockDim);
    gauge_param->gaugeGiB += 2.0*cudaLinkPrecise->bytes/ (1 << 30);
    if (gauge_param->cuda_prec_sloppy != gauge_param->cuda_prec ||
	gauge_param->reconstruct_sloppy != gauge_param->reconstruct) {
	createGaugeField(cudaLinkSloppy, h_gauge, gauge_param->reconstruct_sloppy, 
			 gauge_param->cuda_prec_sloppy, gauge_param->X, gauge_param->anisotropy,
			 gauge_param->blockDim_sloppy);
	gauge_param->gaugeGiB += 2.0*cudaLinkSloppy->bytes/ (1 << 30);
    } else {
	*cudaLinkSloppy = *cudaLinkPrecise;
    }
    
}

void loadFatGaugeQuda(void *h_gauge, QudaGaugeParam *param, void* _cudaLinkPrecise, void* _cudaLinkSloppy)
{
    FullGauge* cudaLinkPrecise = (FullGauge*)_cudaLinkPrecise;
    FullGauge* cudaLinkSloppy = (FullGauge*)_cudaLinkSloppy;
    
    gauge_param = param;
    
    gauge_param->packed_size =18;
    
    createGaugeField(cudaLinkPrecise, h_gauge, QUDA_RECONSTRUCT_NO, 
		     gauge_param->cuda_prec, gauge_param->X, gauge_param->anisotropy, gauge_param->blockDim);
    gauge_param->gaugeGiB += 2.0*cudaLinkPrecise->bytes/ (1 << 30);
    if (gauge_param->cuda_prec_sloppy != gauge_param->cuda_prec ||
	gauge_param->reconstruct_sloppy != gauge_param->reconstruct) {
	createGaugeField(cudaLinkSloppy, h_gauge, QUDA_RECONSTRUCT_NO,
			 gauge_param->cuda_prec_sloppy, gauge_param->X, gauge_param->anisotropy,
			 gauge_param->blockDim_sloppy);
	gauge_param->gaugeGiB += 2.0*cudaLinkSloppy->bytes/ (1 << 30);
    } else {
	*cudaLinkSloppy = *cudaLinkPrecise;
    }
    
}


void endQuda()
{
  freeSpinorBuffer();
  freeGaugeField(&cudaGaugePrecise);
  freeGaugeField(&cudaGaugeSloppy);
}

void checkPrecision(QudaInvertParam *param) {
  if (param->cpu_prec == QUDA_HALF_PRECISION) {
    printf("Half precision not supported on cpu\n");
    exit(-1);
  }
}

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int parity, int dagger)
{
  checkPrecision(inv_param);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);

  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);
  dslashCuda(out, cudaGaugePrecise, in, parity, dagger);
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger)
{
  checkPrecision(inv_param);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);
  MatPCCuda(out, cudaGaugePrecise, in, inv_param->kappa, tmp, inv_param->matpc_type, dagger);
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatPCDagMatPCQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  checkPrecision(inv_param);

  ParitySpinor in = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor out = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  ParitySpinor tmp = allocateParitySpinor(cudaGaugePrecise.X, inv_param->cuda_prec);
  
  loadParitySpinor(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);  
  MatPCDagMatPCCuda(out, cudaGaugePrecise, in, inv_param->kappa, tmp, inv_param->matpc_type);
  retrieveParitySpinor(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeParitySpinor(tmp);
  freeParitySpinor(out);
  freeParitySpinor(in);
}

void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, int dagger) {
  checkPrecision(inv_param);

  FullSpinor in = allocateSpinorField(cudaGaugePrecise.X, inv_param->cuda_prec);
  FullSpinor out = allocateSpinorField(cudaGaugePrecise.X, inv_param->cuda_prec);

  loadSpinorField(in, h_in, inv_param->cpu_prec, inv_param->dirac_order);

  dslashXpayCuda(out.odd, cudaGaugePrecise, in.even, 1, dagger, in.odd, -inv_param->kappa);
  dslashXpayCuda(out.even, cudaGaugePrecise, in.odd, 0, dagger, in.even, -inv_param->kappa);

  retrieveSpinorField(h_out, out, inv_param->cpu_prec, inv_param->dirac_order);

  freeSpinorField(out);
  freeSpinorField(in);
}

void 
invertQuda(void *h_x, void *h_b, QudaInvertParam *param)
{
    invert_param = param;
    
    checkPrecision(param);
    
    int slenh = cudaFatLinkPrecise.volume*spinorSiteSize;
    param->spinorGiB = (double)slenh*((param->cuda_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double): sizeof(float));
    if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
	param->spinorGiB = 1.0*(param->spinorGiB*(param->inv_type == QUDA_CG_INVERTER ? 5 : 7))/(1<<30);
    else
	param->spinorGiB = 1.0*(param->spinorGiB*(param->inv_type == QUDA_CG_INVERTER ? 8 : 9))/(1<<30);


    param->secs = 0;
    param->gflops = 0;
    param->iter = 0;

    double kappa = param->kappa;
    if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa /= cudaFatLinkPrecise.anisotropy;

    FullSpinor b, x;
    ParitySpinor in = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec); // source vector
    ParitySpinor out = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec); // solution vector
    ParitySpinor tmp = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec); // temporary used when applying operator

    if (param->solution_type == QUDA_MAT_SOLUTION) {
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
	    b = allocateSpinorField(cudaFatLinkPrecise.X, invert_param->cuda_prec);
	} else {
	    b.even = out;
	    b.odd = tmp;
	}

	if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) { x.odd = tmp; x.even = out; }
	else { x.even = tmp; x.odd = out; }

	loadSpinorField(b, h_b, param->cpu_prec, param->dirac_order);		
	
	// multiply the source to get the mass normalization
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION) {
	    axCuda(2.0*kappa, b.even);
	    axCuda(2.0*kappa, b.odd);
	}

	// cps uses a different anisotropy normalization
	if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
	    axCuda(1.0/gauge_param->anisotropy, b.even);
	    axCuda(1.0/gauge_param->anisotropy, b.even);
	}

	if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
	    //dslashXpayCuda(in, cudaGaugePrecise, b.odd, 0, 0, b.even, kappa);
	    dslashXpayCuda_st(in, cudaFatLinkPrecise, cudaLongLinkPrecise, b.odd, 0, 0, b.even, kappa);
	} else {
	    //dslashXpayCuda(in, cudaGaugePrecise, b.even, 1, 0, b.odd, kappa);
	    dslashXpayCuda_st(in, cudaFatLinkPrecise, cudaLongLinkPrecise, b.even, 1, 0, b.odd, kappa);
	}

    } else if (param->solution_type == QUDA_MATPC_SOLUTION || 
	       param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION){
	loadParitySpinor(in, h_b, param->cpu_prec, param->dirac_order);

	// multiply the source to get the mass normalization
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION){
	    if (param->solution_type == QUDA_MATPC_SOLUTION){ 
		axCuda(4.0*kappa*kappa, in);
	    }else{
		axCuda(16.0*pow(kappa,4), in);
	    }
	}
	// cps uses a different anisotropy normalization
	if (param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER){
	    if (param->solution_type == QUDA_MATPC_SOLUTION) {
		axCuda(pow(1.0/gauge_param->anisotropy, 2), in);
	    }else{
		axCuda(pow(1.0/gauge_param->anisotropy, 4), in);
	    }
	}
    }

    switch (param->inv_type) {
    case QUDA_CG_INVERTER:
	if (param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
	    copyCuda(out, in);
	    //MatPCCuda(in, cudaGaugePrecise, out, kappa, tmp, param->matpc_type, QUDA_DAG_YES);
	    MatPCCuda_st(in, cudaFatLinkPrecise,cudaLongLinkPrecise, out, kappa, tmp, param->matpc_type, QUDA_DAG_YES);
	}
	invertCgCuda(out, in, cudaFatLinkPrecise, cudaLongLinkPrecise, cudaFatLinkSloppy, cudaLongLinkSloppy, tmp, param);
	break;
    default:
	printf("Inverter type %d not implemented\n", param->inv_type);
	exit(-1);
    }

    if (param->solution_type == QUDA_MAT_SOLUTION) {

	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
	    // qdp dirac fields are even-odd ordered
	    b.even = in;
	    loadSpinorField(b, h_b, param->cpu_prec, param->dirac_order);
	}

	if (param->matpc_type == QUDA_MATPC_EVEN_EVEN) {
	    //dslashXpayCuda(x.odd, cudaGaugePrecise, out, 1, 0, b.odd, kappa);
	    dslashXpayCuda_st(x.odd, cudaFatLinkPrecise, cudaLongLinkPrecise, out, 1, 0, b.odd, kappa);
	} else {
	    //dslashXpayCuda(x.even, cudaGaugePrecise, out, 0, 0, b.even, kappa);
	    dslashXpayCuda_st(x.even, cudaFatLinkPrecise, cudaLongLinkPrecise,out, 0, 0, b.even, kappa);
	}
	
	retrieveSpinorField(h_x, x, param->cpu_prec, param->dirac_order);

	if (param->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeSpinorField(b);

    } else {
	retrieveParitySpinor(h_x, out, param->cpu_prec, param->dirac_order);
    }

    freeParitySpinor(tmp);
    freeParitySpinor(in);
    freeParitySpinor(out);

    return;
}

int
invertQuda_milc(void *h_x, void *h_b, QudaInvertParam *param, double mass, QudaParity parity)
{
    int iters = 0;
    
    invert_param = param;
    
    checkPrecision(param);
    
    int slenh = cudaFatLinkPrecise.volume*spinorSiteSize;
    param->spinorGiB = (double)slenh*((param->cuda_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double): sizeof(float));
    if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
	param->spinorGiB = 1.0*(param->spinorGiB*(param->inv_type == QUDA_CG_INVERTER ? 5 : 7))/(1<<30);
    else
	param->spinorGiB = 1.0*(param->spinorGiB*(param->inv_type == QUDA_CG_INVERTER ? 8 : 9))/(1<<30);

    
    param->secs = 0;
    param->gflops = 0;
    param->iter = 0;
    
    int orig_parity = parity;
    if (parity ==QUDA_EVENODD){
	parity=QUDA_EVEN;
    }  
    

    ParitySpinor in = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec); 
    ParitySpinor out = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec);
    ParitySpinor tmp = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec);
    
 start:	
    loadParitySpinor(in, h_b, param->cpu_prec, param->dirac_order);    
    loadParitySpinor(out, h_x, param->cpu_prec, param->dirac_order);    
    iters += invertCgCuda_milc_parity(out, in, cudaFatLinkPrecise, cudaLongLinkPrecise, cudaFatLinkSloppy, cudaLongLinkSloppy, tmp, param, mass, parity);
    retrieveParitySpinor(h_x, out, param->cpu_prec, param->dirac_order);
    
    if(orig_parity == QUDA_EVENODD){
	parity =QUDA_ODD;
	size_t sSize = (param->cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);	    
	h_x = ((char*)h_x) + Vh*spinorSiteSize*sSize;
	h_b = ((char*)h_b)+ Vh*spinorSiteSize*sSize;
	orig_parity=QUDA_EVEN; //avoid infinite loop
	goto start;
    }
    

    freeParitySpinor(tmp);
    freeParitySpinor(in);
    freeParitySpinor(out);
    
    return iters;
}


int
invertQuda_milc_multi_offset(void **_h_x, void *_h_b, QudaInvertParam *param,
			     double* offsets, int num_offsets, QudaParity parity, double* residue_sq)
{

    if (num_offsets <= 0) {
	return 0;
    }

    int iters = 0;    
    invert_param = param;    
    checkPrecision(param);    
    param->secs = 0;
    param->gflops = 0;
    param->iter = 0;
    
    double low_offset = offsets[0];
    int low_index = 0;
    for (int i=1;i < num_offsets;i++){
	if (offsets[i] < low_offset){
	    low_offset = offsets[i];
	    low_index = i;
	}
    }
	
    void* h_x[num_offsets];
    void* h_b = _h_b;
    for(int i=0;i < num_offsets;i++){
	h_x[i] = _h_x[i];
    }
	
    if (low_index != 0){
	void* tmp = h_x[0];
	h_x[0] = h_x[low_index] ;
	h_x[low_index] = tmp;
	
	double tmp1 = offsets[0];
	offsets[0]= offsets[low_index];
	offsets[low_index] =tmp1;
    }
	
    int orig_parity = parity;
    if (parity ==QUDA_EVENODD){
	parity=QUDA_EVEN;
    }
    ParitySpinor in = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec); 
    ParitySpinor out[num_offsets];
    for(int i =0;i < num_offsets; i++){
	out[i]= allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec);
    }
    ParitySpinor tmp = allocateParitySpinor(cudaFatLinkPrecise.X, invert_param->cuda_prec);
	
 start:	
    loadParitySpinor(in, h_b, param->cpu_prec, param->dirac_order);    
    iters += invertCgCuda_milc_multi_mass_parity(out, in, cudaFatLinkPrecise, cudaLongLinkPrecise, 
						 cudaFatLinkSloppy, cudaLongLinkSloppy, tmp, param, offsets, num_offsets, parity, residue_sq);
    
    for(int i =0; i < num_offsets; i++){
	retrieveParitySpinor(h_x[i], out[i], param->cpu_prec, param->dirac_order);
    }
	
    if(orig_parity == QUDA_EVENODD){
	parity =QUDA_ODD;
	size_t sSize = (param->cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);	    
	for(int i=0;i < num_offsets;i++){
	    h_x[i] = ((char*)(h_x[i])) + Vh*spinorSiteSize*sSize;
	}
	h_b = ((char*)h_b)+ Vh*spinorSiteSize*sSize;
	orig_parity=QUDA_EVEN; //avoid infinite loop
	goto start;
    }
	

    freeParitySpinor(tmp);
    freeParitySpinor(in);
    for(int i=0;i < num_offsets;i++){
	freeParitySpinor(out[i]);	
    }

    
    PRINTF("total iters=%d\n", iters);
    return iters;
}


