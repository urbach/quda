#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <quda.h>
#include <util_quda.h>
#include <dslash_reference.h>
#include "misc.h"
#include <string.h>

QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaPrecision spinor_prec = QUDA_SINGLE_PRECISION;
QudaPrecision  link_prec = QUDA_SINGLE_PRECISION;
QudaPrecision  cpu_prec = QUDA_DOUBLE_PRECISION;

QudaReconstructType link_recon_sloppy = QUDA_RECONSTRUCT_NOT_SET;
QudaPrecision spinor_prec_sloppy = QUDA_PRECISION_NOT_SET;
QudaPrecision  link_prec_sloppy = QUDA_PRECISION_NOT_SET;
static int testtype = 0;
static int tdim =24;
static int sdim = 8;
int
invert_test(void)
{
  int device = 0;

  //void *gauge[4];
  void *fatlink[4];
  void *longlink[4];
  
  QudaGaugeParam Gauge_param;
  QudaInvertParam inv_param;

  Gauge_param.X[0] = sdim;
  Gauge_param.X[1] = sdim;
  Gauge_param.X[2] = sdim;
  Gauge_param.X[3] = tdim;
  setDims(Gauge_param.X);

  Gauge_param.blockDim = 64;
  Gauge_param.blockDim_sloppy = 64;

  Gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;

  Gauge_param.cuda_prec = link_prec;
  Gauge_param.reconstruct = link_recon;

  Gauge_param.cuda_prec_sloppy = link_prec_sloppy;
  Gauge_param.reconstruct_sloppy = link_recon_sloppy;
  
  Gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  Gauge_param.anisotropy = 1.0;

  inv_param.inv_type = QUDA_CG_INVERTER;

  Gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  Gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param = &Gauge_param;
  
  double mass = -0.95;
  inv_param.kappa = 1.0 / (2.0*(4 + mass));
  inv_param.tol = 1e-12;
  inv_param.maxiter = 10000;
  inv_param.reliable_delta = 1e-3;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = spinor_prec;
  inv_param.cuda_prec_sloppy = spinor_prec_sloppy;
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;  // preservation doesn't work with reliable?
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  size_t gSize = (Gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
      //gauge[dir] = malloc(V*gaugeSiteSize*gSize);
      fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
      longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  //  construct_gauge_field(gauge, 1, Gauge_param.cpu_prec);
  construct_fat_long_gauge_field(fatlink, longlink, 1, Gauge_param.cpu_prec);
  
  void *spinorIn = malloc(V*spinorSiteSize*sSize);
  void *spinorOut = malloc(V*spinorSiteSize*sSize);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize);

  int i0 = 0;
  int s0 = 0;
  int c0 = 0;
  construct_spinor_field(spinorIn, 1, i0, s0, c0, inv_param.cpu_prec);


  initQuda(device);
  //loadGaugeQuda((void*)gauge, &Gauge_param);
  double time0 = -((double)clock()); // Start the timer
  loadFatGaugeQuda(fatlink, &Gauge_param, &cudaFatLinkPrecise, &cudaFatLinkSloppy);
  loadGaugeQuda_st(longlink, &Gauge_param, &cudaLongLinkPrecise, &cudaLongLinkSloppy);
  
  invertQuda(spinorOut, spinorIn, &inv_param);

  time0 += clock(); // stop the timer
  time0 /= CLOCKS_PER_SEC;

  printf("Cuda Space Required. Spinor:%f + Gauge:%f GiB\n", 
	 inv_param.spinorGiB, Gauge_param.gaugeGiB);
  printf("done: %i iter / %g secs = %g gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  mat(spinorCheck, fatlink, longlink, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, Gauge_param.cpu_prec);
  if  (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
    ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);

  mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
  printf("Relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  endQuda();

  return 0;
}


int
invert_milc_test(void)
{
    int device = 0;
    
    void *fatlink[4];
    void *longlink[4];
  
    QudaGaugeParam Gauge_param;
    QudaInvertParam inv_param;

    Gauge_param.X[0] = sdim;
    Gauge_param.X[1] = sdim;
    Gauge_param.X[2] = sdim;
    Gauge_param.X[3] = tdim;
    setDims(Gauge_param.X);
    
    Gauge_param.blockDim = 64;
    Gauge_param.blockDim_sloppy = 64;

    Gauge_param.cpu_prec = cpu_prec;

    Gauge_param.cuda_prec = link_prec;
    Gauge_param.reconstruct = link_recon;

    Gauge_param.cuda_prec_sloppy = link_prec_sloppy;
    Gauge_param.reconstruct_sloppy = link_recon_sloppy;
  
    Gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

    Gauge_param.anisotropy = 1.0;

    inv_param.inv_type = QUDA_CG_INVERTER;

    Gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
    Gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gauge_param = &Gauge_param;
    
    double mass = -0.95;
    inv_param.kappa = 1.0 / (2.0*(4 + mass));
    inv_param.tol = 1e-12;
    inv_param.maxiter = 10000;
    inv_param.reliable_delta = 1e-3;
    inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;
    inv_param.cpu_prec = cpu_prec;
    inv_param.cuda_prec = spinor_prec;
    inv_param.cuda_prec_sloppy = spinor_prec_sloppy;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
    inv_param.dirac_order = QUDA_DIRAC_ORDER;

    size_t gSize = (Gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

    for (int dir = 0; dir < 4; dir++) {
	fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
	longlink[dir] = malloc(V*gaugeSiteSize*gSize);
    }
    construct_fat_long_gauge_field(fatlink, longlink, 1, Gauge_param.cpu_prec);

    for (int dir = 0; dir < 4; dir++) {
	for(int i = 0;i < V*gaugeSiteSize;i++){
	    if (Gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION){
		((double*)fatlink[dir])[i] = 0.5 *rand()/RAND_MAX;
	    }else{
		((float*)fatlink[dir])[i] = 0.5* rand()/RAND_MAX;
	    }
	}
    }
    
    void *spinorIn = malloc(V*spinorSiteSize*sSize);
    void *spinorOut = malloc(V*spinorSiteSize*sSize);
    void *spinorCheck = malloc(V*spinorSiteSize*sSize);
    void *tmp = malloc(V*spinorSiteSize*sSize);
    
    memset(spinorIn, 0, V*spinorSiteSize*sSize);
    memset(spinorOut, 0, V*spinorSiteSize*sSize);
    memset(spinorCheck, 0, V*spinorSiteSize*sSize);
    memset(tmp, 0, V*spinorSiteSize*sSize);

    void* spinorInEven = spinorIn;
    void* spinorInOdd = ((char*)spinorIn) + Vh*spinorSiteSize*sSize;
    void* spinorOutEven = spinorOut;
    void* spinorOutOdd  = ((char*)spinorOut)+ Vh*spinorSiteSize*sSize;
    void* spinorCheckEven = spinorCheck;
    void* spinorCheckOdd  = ((char*)spinorCheck)+ Vh*spinorSiteSize*sSize;
    
    int i0 = 0;
    int s0 = 0;
    int c0 = 0;
    construct_spinor_field(spinorIn, 1, i0, s0, c0, inv_param.cpu_prec);
    
    
    initQuda(device);

    double time0 = -((double)clock()); // Start the timer

    loadFatGaugeQuda(fatlink, &Gauge_param, &cudaFatLinkPrecise, &cudaFatLinkSloppy);
    loadGaugeQuda_st(longlink, &Gauge_param, &cudaLongLinkPrecise, &cudaLongLinkSloppy);
    
    unsigned long volume=V;
    unsigned long nflops=2*1187; //from MILC's CG routine
    double nrm2=0;
    double src2=0;
    switch(testtype){

    case 0: //even
	volume = Vh;

	invertQuda_milc(spinorOutEven, spinorInEven, &inv_param, mass, QUDA_EVEN);
	
	time0 += clock(); 
	time0 /= CLOCKS_PER_SEC;
	
	matdagmat_milc(spinorCheckEven, fatlink, longlink, spinorOutEven, mass, 0, inv_param.cpu_prec, Gauge_param.cpu_prec, tmp, QUDA_EVEN);
	if  (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
	    ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	
	mxpy(spinorInEven, spinorCheckEven, Vh*spinorSiteSize, inv_param.cpu_prec);
	nrm2 = norm_2(spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
	src2 = norm_2(spinorIn, Vh*spinorSiteSize, inv_param.cpu_prec);
	break;
	
    case 1: //odd
	
	volume = Vh;
	invertQuda_milc(spinorOutOdd, spinorInOdd, &inv_param, mass, QUDA_ODD);	
	time0 += clock(); // stop the timer
	time0 /= CLOCKS_PER_SEC;
	
	
	matdagmat_milc(spinorCheckOdd, fatlink, longlink, spinorOutOdd, mass, 0, inv_param.cpu_prec, Gauge_param.cpu_prec, tmp, QUDA_ODD);	
	mxpy(spinorInOdd, spinorCheckOdd, Vh*spinorSiteSize, inv_param.cpu_prec);
	nrm2 = norm_2(spinorCheckOdd, Vh*spinorSiteSize, inv_param.cpu_prec);
	src2 = norm_2(spinorInOdd, Vh*spinorSiteSize, inv_param.cpu_prec);
	
	break;
    case 2: //full spinor

	volume = Vh; //FIXME: the time reported is only parity time
	invertQuda_milc(spinorOut, spinorIn, &inv_param, mass, QUDA_EVENODD);
	
	time0 += clock(); // stop the timer
	time0 /= CLOCKS_PER_SEC;
	
	matdagmat_milc(spinorCheck, fatlink, longlink, spinorOut, mass, 0, inv_param.cpu_prec, Gauge_param.cpu_prec, tmp, QUDA_EVENODD);

	mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);

	break;
    case 3: //multi mass CG, even
    case 4:
    case 5:
#define NUM_OFFSETS 4
	
	nflops = 2*(1205 + 15* NUM_OFFSETS); //from MILC's multimass CG routine
	double masses[NUM_OFFSETS] ={1.05, 1.23, 2.64, 2.33};
	    double offsets[NUM_OFFSETS];	
	    //int num_offsets =NUM_OFFSETS;
	    int num_offsets =  4;
	    void* spinorOutArray[NUM_OFFSETS];
	    void* in;
	    int parity;
	    int len;

	    for (int i=0; i< num_offsets;i++){
		offsets[i] = 4*masses[i]*masses[i];
	    }
	    
	    if (testtype == 3){
		parity = QUDA_EVEN;
		in=spinorInEven;
		len=Vh;
		volume = Vh;
		
		spinorOutArray[0] = spinorOutEven;
		for (int i=1; i< num_offsets;i++){
		    spinorOutArray[i] = malloc(Vh*spinorSiteSize*sSize);
		}		
	    }else if (testtype ==4){
		parity = QUDA_ODD;
		in=spinorInOdd;
		len = Vh;
		volume = Vh;

		spinorOutArray[0] = spinorOutOdd;
		for (int i=1; i< num_offsets;i++){
		    spinorOutArray[i] = malloc(Vh*spinorSiteSize*sSize);
		}
	    }else {
		parity = QUDA_EVENODD;
		in=spinorIn;
		len= V;
		volume = Vh; //FIXME: the time reported is only parity time

		spinorOutArray[0] = spinorOut;
		for (int i=1; i< num_offsets;i++){
		    spinorOutArray[i] = malloc(V*spinorSiteSize*sSize);
		}		
	    }
	    
	    
	    double residue_sq;
	    invertQuda_milc_multi_offset(spinorOutArray, in, &inv_param, offsets, num_offsets, parity, &residue_sq);	
	    cudaThreadSynchronize();
	    printf("Final residue squred =%g\n", residue_sq);
	    time0 += clock(); // stop the timer
	    time0 /= CLOCKS_PER_SEC;

	    printf("done: total time = %g secs, %i iter / %g secs = %g gflops, \n", 
		   time0, inv_param.iter, inv_param.secs,
		   1e-9*nflops*volume*(inv_param.iter+inv_param.rUpdate)/inv_param.secs);
	    
	    printf("checking the solution\n");
	    for(int i=0;i < num_offsets;i++){
		printf("%dth solution: ", i);
		matdagmat_milc(spinorCheck, fatlink, longlink, spinorOutArray[i], masses[i], 0, inv_param.cpu_prec, Gauge_param.cpu_prec, tmp, parity);
		mxpy(in, spinorCheck, len*spinorSiteSize, inv_param.cpu_prec);
		double nrm2 = norm_2(spinorCheck, len*spinorSiteSize, inv_param.cpu_prec);
		double src2 = norm_2(in, len*spinorSiteSize, inv_param.cpu_prec);
		printf("relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));
	    }
	    
	    for(int i=1; i < num_offsets;i++){
		free(spinorOutArray[i]);
	    }
	
    }//switch
    
    if (testtype <=2){
	printf("Relative residual, requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));
	
	printf("done: total time = %g secs, %i iter / %g secs = %g gflops, \n", 
	       time0, inv_param.iter, inv_param.secs,
	       1.0e-9*nflops*volume*(inv_param.iter+inv_param.rUpdate)/(inv_param.secs));
    }
    endQuda();

    if (tmp){
	free(tmp);
    }
    return 0;
}




void
display_test_info()
{
    printf("running the following test:\n");
    
    printf("spinor_prec \t link_prec \tlink_recon\tsloppy_spinor_prec \t sloppy_link_prec \t sloppy_link_recon\ttest_type\t S_dimension T_dimension\n");
    printf("%s \t\t  %s   \t%s\t\t%s\t\t\t\t%s\t\t\t%s\t\t%s\t\t%d\t\t%d\n", get_prec_str(spinor_prec),
	   get_prec_str(link_prec), get_recon_str(link_recon), get_prec_str(spinor_prec_sloppy),
	   get_prec_str(link_prec_sloppy), get_recon_str(link_recon_sloppy), get_test_type(testtype), sdim, tdim);     
    return ;
    
}

void
usage(char** argv )
{
    printf("Usage: %s <args>\n", argv[0]);
    printf("--sprec <double/single/half> \t Spinor precision\n"); 
    printf("--gprec <double/single/half> \t Link precision\n"); 
    printf("--recon <8/12> \t\t\t Long link reconstruction type\n"); 
    printf("--type <0/1/2/3/4/5> \t\t Testing type(0=even, 1=odd, 2=full, 3=multimass even, 4=multimass odd, 5=multimass full)\n"); 
    printf("--tdim <n>     \t\t\t T dimension\n");
    printf("--help \t\t\t\t Print out this message\n"); 
    exit(1);
    return ;
}


int main(int argc, char** argv)
{

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

        if( strcmp(argv[i], "--sprec_sloppy") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    spinor_prec_sloppy =  get_prec(argv[i+1]);
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
	
	if( strcmp(argv[i], "--gprec_sloppy") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    link_prec_sloppy =  get_prec(argv[i+1]);
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
	
	if( strcmp(argv[i], "--recon_sloppy") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    link_recon_sloppy =  get_recon(argv[i+1]);
            i++;
            continue;	    
        }
	
	if( strcmp(argv[i], "--type") == 0){
            if (i+1 >= argc){
                usage(argv);
            }	    
	    testtype = atoi(argv[i+1]);
            i++;
            continue;	    
        }

        if( strcmp(argv[i], "--cprec") == 0){
            if (i+1 >= argc){
                usage(argv);
            }
            cpu_prec= get_prec(argv[i+1]);
            i++;
            continue;
        }

        if( strcmp(argv[i], "--tdim") == 0){
            if (i+1 >= argc){
                usage(argv);
            }
            tdim= atoi(argv[i+1]);
	    if (tdim < 0 || tdim > 128){
		printf("ERROR: invalid T dimention (%d)\n", tdim);
		usage(argv);
	    }
            i++;
            continue;
        }		
       if( strcmp(argv[i], "--sdim") == 0){
            if (i+1 >= argc){
                usage(argv);
            }
            sdim= atoi(argv[i+1]);
            if (sdim < 0 || sdim > 128){
                printf("ERROR: invalid S dimention (%d)\n", sdim);
                usage(argv);
            }
            i++;
            continue;
        }


        fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
        usage(argv);
    }


    if (spinor_prec_sloppy == QUDA_PRECISION_NOT_SET){
	spinor_prec_sloppy = spinor_prec;
    }
    if (link_prec_sloppy == QUDA_PRECISION_NOT_SET){
	link_prec_sloppy = link_prec;
    }
    if (link_recon_sloppy == QUDA_RECONSTRUCT_NOT_SET){
	link_recon_sloppy = link_recon;
    }
    
    display_test_info();
    
    //srand(1090);

    //invert_test();
    
    invert_milc_test();
    

    return 0;
}
