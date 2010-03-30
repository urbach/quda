#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <dslash_reference.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>
#include <string.h>
#include "misc.h"


// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 0;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

FullGauge gauge;
FullGauge cudaFatLink;
FullGauge cudaLongLink;

FullSpinor cudaSpinor;
FullSpinor cudaSpinorOut;
ParitySpinor tmp;

void *hostGauge[4];
void *fatlink[4], *longlink[4];

void *spinor, *spinorEven, *spinorOdd;
void *spinorRef, *spinorRefEven, *spinorRefOdd;
void *spinorGPU, *spinorGPUEven, *spinorGPUOdd;
    
double kappa = 1.0;
int ODD_BIT = 1;
int dagger_bit = 0;
int TRANSFER = 0; // include transfer time in the benchmark?
int tdim = 24;
int sdim = 8;

QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaPrecision spinor_prec = QUDA_SINGLE_PRECISION;
QudaPrecision  link_prec = QUDA_SINGLE_PRECISION;

void
init()
{    
    gaugeParam.X[0] = sdim;
    gaugeParam.X[1] = sdim;
    gaugeParam.X[2] = sdim;
    gaugeParam.X[3] = tdim;

    setDims(gaugeParam.X);

    gaugeParam.blockDim = 64;

    gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
    gaugeParam.cuda_prec = link_prec;
    gaugeParam.reconstruct = link_recon;
    gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
    gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;
    
    gaugeParam.anisotropy = 2.3;
    gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
    gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
    gaugeParam.gaugeGiB =0;
    gauge_param = &gaugeParam;
    
    inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec = spinor_prec;
    if (test_type == 2) inv_param.dirac_order = QUDA_DIRAC_ORDER;
    else inv_param.dirac_order = QUDA_DIRAC_ORDER;
    inv_param.kappa = kappa;
    invert_param = &inv_param;

    size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

    for (int dir = 0; dir < 4; dir++) {
	fatlink[dir] = malloc(V*gaugeSiteSize* gSize);
	longlink[dir] = malloc(V*gaugeSiteSize* gSize);
    }
    if (fatlink == NULL || longlink == NULL){
	fprintf(stderr, "ERROR: malloc failed for fatlink/longlink\n");
	exit(1);
    }
  
  
    spinor = malloc(V*spinorSiteSize*sSize);
    spinorRef = malloc(V*spinorSiteSize*sSize);
    spinorGPU = malloc(V*spinorSiteSize*sSize);
    spinorEven = spinor;
    spinorRefEven = spinorRef;
    spinorGPUEven = spinorGPU;
    if (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) {
	spinorOdd = (void*)((double*)spinor + Vh*spinorSiteSize);
	spinorRefOdd = (void*)((double*)spinorRef + Vh*spinorSiteSize);
	spinorGPUOdd = (void*)((double*)spinorGPU + Vh*spinorSiteSize);
    } else {
	spinorOdd = (void*)((float*)spinor + Vh*spinorSiteSize);
	spinorRefOdd = (void*)((float*)spinorRef + Vh*spinorSiteSize);
	spinorGPUOdd = (void*)((float*)spinorGPU + Vh*spinorSiteSize);
    }

    printf("Randomizing fields...");
    
    construct_fat_long_gauge_field(fatlink, longlink, 1, gaugeParam.cpu_prec);
    
    construct_spinor_field(spinor, 1, 0, 0, 0, inv_param.cpu_prec);
    
    printf("done.\n"); fflush(stdout);

#if 1
    //printf("links are:\n");
    //display_link(fatlink[0], 1, gaugeParam.cpu_prec);
    //display_link(longlink[0], 1, gaugeParam.cpu_prec);
    
    for (int i =0;i < 4 ;i++){
	int dir = 2*i;
	link_sanity_check(fatlink[i], V, gaugeParam.cpu_prec, dir, &gaugeParam);
	link_sanity_check(longlink[i], V, gaugeParam.cpu_prec, dir, &gaugeParam);
    }

    //printf("spinors are:\n");  
    //display_spinor(spinor, 10, inv_param.cpu_prec);
#endif

    int dev = 0;
    initQuda(dev);

    loadFatGaugeQuda(fatlink, &gaugeParam, &cudaFatLinkPrecise, &cudaFatLinkSloppy);
    loadGaugeQuda_st(longlink, &gaugeParam, &cudaLongLinkPrecise, &cudaLongLinkSloppy);
    
    //gauge = cudaFatLinkPrecise;
    cudaFatLink = cudaFatLinkPrecise;
    cudaLongLink = cudaLongLinkPrecise;
    
    printf("Sending fields to GPU..."); fflush(stdout);
    
    if (!TRANSFER) {

	gaugeParam.X[0] /= 2;
	tmp = allocateParitySpinor(gaugeParam.X, inv_param.cuda_prec);
	cudaSpinor = allocateSpinorField(gaugeParam.X, inv_param.cuda_prec);
	cudaSpinorOut = allocateSpinorField(gaugeParam.X, inv_param.cuda_prec);
	gaugeParam.X[0] *= 2;
	
	printf("spinor allocation done\n");
	if (test_type < 2) {
	    loadParitySpinor(cudaSpinor.even, spinorEven, inv_param.cpu_prec, 
			     inv_param.dirac_order);
	} else {
	    loadSpinorField(cudaSpinor, spinor, inv_param.cpu_prec, 
			    inv_param.dirac_order);
	}

    }


    return;
}

void end() 
{
    // release memory
    for (int dir = 0; dir < 4; dir++) {
	free(fatlink[dir]);
	free(longlink[dir]);
    }
    free(spinorGPU);
    free(spinor);
    free(spinorRef);
    if (!TRANSFER) {
	freeSpinorField(cudaSpinorOut);
	freeSpinorField(cudaSpinor);
	freeParitySpinor(tmp);
    }
    endQuda();
}

double dslashCUDA() {
    
    // execute kernel
    const int LOOPS = 100;
    printf("Executing %d kernel loops...", LOOPS);
    fflush(stdout);
    stopwatchStart();
    for (int i = 0; i < LOOPS; i++) {
	switch (test_type) {
	case 0:
	    if (TRANSFER){
		dslashQuda(spinorOdd, spinorEven, &inv_param, ODD_BIT, dagger_bit);
	    }
	    else {
		dslashCuda_st(cudaSpinor.odd, cudaFatLink, cudaLongLink, cudaSpinor.even, ODD_BIT, dagger_bit);
	    }	   
	    break;
	case 1:
	    if (TRANSFER){
		MatPCQuda(spinorOdd, spinorEven, &inv_param, dagger_bit);
	    }else {
		MatPCCuda_st(cudaSpinor.odd, cudaFatLink, cudaLongLink, cudaSpinor.even, kappa, tmp, QUDA_MATPC_EVEN_EVEN, dagger_bit);
	    }
	    break;
	case 2:
	    if (TRANSFER) MatQuda(spinorGPU, spinor, &inv_param, dagger_bit);
	    else MatCuda_st(cudaSpinorOut, cudaFatLink, cudaLongLink, cudaSpinor, kappa, dagger_bit);
	}
    }
    
    // check for errors
    cudaError_t stat = cudaGetLastError();
    if (stat != cudaSuccess)
	printf("with ERROR: %s\n", cudaGetErrorString(stat));

    cudaThreadSynchronize();
    double secs = stopwatchReadSeconds() / LOOPS;
    printf("done.\n\n");

    return secs;
}

void 
dslashRef()
{
    
    // compare to dslash reference implementation
    printf("Calculating reference implementation...");
    fflush(stdout);
    switch (test_type) {
    case 0:
	dslash(spinorRef, fatlink, longlink, spinorEven, ODD_BIT, dagger_bit, 
		  inv_param.cpu_prec, gaugeParam.cpu_prec);
	break;
    case 1:    
	matpc(spinorRef, fatlink, longlink, spinorEven, kappa, QUDA_MATPC_EVEN_EVEN, dagger_bit, 
		 inv_param.cpu_prec, gaugeParam.cpu_prec);
	break;
    case 2:
	mat(spinorRef, fatlink, longlink, spinor, kappa, dagger_bit, 
	       inv_param.cpu_prec, gaugeParam.cpu_prec);
	break;
    default:
	printf("Test type not defined\n");
	exit(-1);
    }
    
    printf("done.\n");
    
}

void 
dslashTest() 
{
    
    init();
    
    float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
    float sharedKB = (float)dslashCudaSharedBytes(inv_param.cuda_prec, gaugeParam.blockDim) / (1 << 10);
    printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
    printf("Gauge mem: %.3f GiB\n", gaugeParam.gaugeGiB);
    printf("Shared mem: %.3f KB\n", sharedKB);

    int attempts = 1;

    dslashRef();
    
    for (int i=0; i<attempts; i++) {
	
	double secs = dslashCUDA();
	cudaThreadSynchronize(); CUERR;
		
	if (!TRANSFER) {
	    if (test_type < 2) 
		retrieveParitySpinor(spinorOdd, cudaSpinor.odd, inv_param.cpu_prec, inv_param.dirac_order);
	    else 
		retrieveSpinorField(spinorGPU, cudaSpinorOut, inv_param.cpu_prec, inv_param.dirac_order);
	}
	
	printf("\n%fms per loop\n", 1000*secs);
	 
	//int flops = test_type ? 1320*2 + 48 : 1320;
	//int floats = test_type ? 2*(7*24+8*gaugeParam.packed_size+24)+24 : 7*24+8*gaugeParam.packed_size+24;
	int flops = test_type ? 1146*2 + 12: 1146;
	int link_floats = 8*gaugeParam.packed_size+8*18;
	int spinor_floats = 8*6*2 + 6;
	int link_float_size = 0;
	int spinor_float_size = 0;
	switch(gaugeParam.cuda_prec){
	case QUDA_DOUBLE_PRECISION:
	    link_float_size = sizeof(double);
	    break;
	case QUDA_SINGLE_PRECISION:
	    link_float_size = sizeof(float);
	    break;
	case QUDA_HALF_PRECISION:
	    link_float_size = sizeof(float)/2;
	    break;
	default:
	    printf("ERROR: invalid precison type\n");
	    break;
	}
	switch(inv_param.cuda_prec){
	case QUDA_DOUBLE_PRECISION:
	    spinor_float_size = sizeof(double);
	    break;
	case QUDA_SINGLE_PRECISION:
	    spinor_float_size = sizeof(float);
	    break;
	case QUDA_HALF_PRECISION:
	    spinor_float_size = sizeof(float)/2;
	    break;
	default:
	    printf("ERROR: invalid precison type\n");
	    break;
	}


	link_floats = test_type? (2*link_floats): link_floats;
	spinor_floats = test_type? (2*spinor_floats): spinor_floats;
	
	int bytes_for_one_site = link_floats* link_float_size + spinor_floats * spinor_float_size;
	
	printf("GFLOPS = %f\n", 1.0e-9*flops*Vh/secs);
	printf("GiB/s = %f\n\n", Vh*bytes_for_one_site/(secs*(1<<30)));
       

	double tmp=0;
	double* s = (double*)spinorOdd;
	for (int j=0; j < Vh*spinorSiteSize; j++){
	    tmp += s[j]*s[j];	    
	    if (s[j]!= s[j]){
		printf("ERROR:not a number, s[%d]=%g\n", i, s[i]);
		exit(1);
	    }
	}
	
	int res;
	if (test_type < 2) res = compare_floats(spinorOdd, spinorRef, Vh*spinorSiteSize, 1e-3, inv_param.cpu_prec);
	else res = compare_floats(spinorGPU, spinorRef, V*spinorSiteSize, 1e-3, inv_param.cpu_prec);
	
#if 1      
	if (test_type < 2) strong_check(spinorRef, spinorOdd, Vh, inv_param.cpu_prec);
	else strong_check(spinorRef, spinorGPU, V, inv_param.cpu_prec);

#endif    
	
	printf("%d Test %s\n", i, (1 == res) ? "PASSED" : "FAILED");	    
    }  
    
    
    end();
  
}


void
display_test_info()
{
    printf("running the following test:\n");
 
    printf("spinor_precision \t link_precision \tlink_reconstruct     test_type     dagger_bit\t   S_dimension \tT_dimension\n");
    printf("\t%s \t\t\t%s \t\t\t%s \t\t%d \t\t%d\t\t%d\t\t%d \n", get_prec_str(spinor_prec),
	   get_prec_str(link_prec), get_recon_str(link_recon), 
	   test_type, dagger_bit, sdim, tdim);
    return ;
    
}

void
usage(char** argv )
{
    printf("Usage: %s <args>\n", argv[0]);
    printf("--sprec <double/single/half> \t Spinor precision\n"); 
    printf("--gprec <double/single/half> \t Link precision\n"); 
    printf("--recon <8/12> \t\t\t Long link reconstruction type\n"); 
    printf("--type <0/1/2> \t\t\t Test type\n"); 
    printf("--dagger \t\t\t Set the dagger to 1\n"); 
    printf("--tdim \t\t\t\t Set T dimention size(default 24)\n");     
    printf("--sdim \t\t\t\t Set space dimention size\n"); 
    printf("--help \t\t\t\t Print out this message\n"); 
    exit(1);
    return ;
}

int 
main(int argc, char **argv) 
{
    /*
      if(argc >=2){
      test_type = atoi(argv[1]);
      }
    if(argc >=3){
	dagger_bit = atoi(argv[2]);
    }
    printf("test type=%d, dagger_bit=%d\n", test_type, dagger_bit);
    */
    
    int i;
    for (i =1;i < argc; i++){
	
        if( strcmp(argv[i], "--help")== 0){
            usage(argv);
        }
	
        if( strcmp(argv[i], "--sprec") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    spinor_prec =  get_prec(argv[i+1]);
            i++;
            continue;	    
        }
	
	if( strcmp(argv[i], "--gprec") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    link_prec =  get_prec(argv[i+1]);
            i++;
            continue;	    
        }
	
	if( strcmp(argv[i], "--recon") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    link_recon =  get_recon(argv[i+1]);
            i++;
            continue;	    
        }
	
	if( strcmp(argv[i], "--type") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    test_type =  atoi(argv[i+1]);
	    if (test_type < 0 || test_type > 2){
		fprintf(stderr, "Error: invalid test type\n");
		exit(1);
	    }
            i++;
            continue;	    
        }

	if( strcmp(argv[i], "--tdim") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    tdim =  atoi(argv[i+1]);
	    if (tdim < 0 || tdim > 128){
		fprintf(stderr, "Error: invalid t dimention\n");
		exit(1);
	    }
            i++;
            continue;	    
        }

	if( strcmp(argv[i], "--sdim") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    sdim =  atoi(argv[i+1]);
	    if (sdim < 0 || sdim > 128){
		fprintf(stderr, "Error: invalid S dimention\n");
		exit(1);
	    }
            i++;
            continue;	    
        }
	
	if( strcmp(argv[i], "--dagger") == 0){
	    dagger_bit = 1;
            continue;	    
        }	

        fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
        usage(argv);
    }

    display_test_info();

    dslashTest();
}
