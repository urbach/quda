#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda.h>
#include <util_quda.h>
#include <spinor_quda.h>
#include <gauge_quda.h>
#include <sys/time.h>
#include "misc.h"

int verbose=1;

int
invertCgCuda(ParitySpinor x, ParitySpinor source, FullGauge fatlinkPrecise, FullGauge longlinkPrecise,
	     FullGauge fatlinkSloppy, FullGauge longlinkSloppy, ParitySpinor tmp, QudaInvertParam *perf)
{
    ParitySpinor p = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    ParitySpinor Ap = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    ParitySpinor y = allocateParitySpinor(x.X, x.precision);
    ParitySpinor r = allocateParitySpinor(x.X, x.precision);
    
    ParitySpinor b;
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
	b = allocateParitySpinor(x.X, x.precision);
	copyCuda(b, source);
    } else {
	b = source;
    }

    ParitySpinor x_sloppy, r_sloppy, tmp_sloppy;
    
    if (invert_param->cuda_prec_sloppy == x.precision) {
	x_sloppy = x;
	r_sloppy = r;
	tmp_sloppy = tmp;
    } else {
	x_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
	r_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
	tmp_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    }
    
    copyCuda(r, b);
    if (r_sloppy.precision != r.precision) copyCuda(r_sloppy, r);
    copyCuda(p, r_sloppy);
    zeroCuda(x_sloppy);
    zeroCuda(y);

    double b2 = 0.0;
    b2 = normCuda(b);

    double r2 = b2;
    double r2_old;
    double stop = r2*perf->tol*perf->tol; // stopping condition of solver
    
    double alpha, beta;
    double pAp;
    
    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = invert_param->reliable_delta;
    
    int k=0;
    int xUpdate = 0, rUpdate = 0;
    
    PRINTF("%d iterations, r2 = %e\n", k, r2);
    stopwatchStart();
    while (r2 > stop && k<perf->maxiter) {
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	MatPCDagMatPCCuda_st(Ap, fatlinkSloppy, longlinkSloppy, p, perf->kappa, tmp_sloppy, perf->matpc_type);
      
	pAp = reDotProductCuda(p, Ap);

	alpha = r2 / pAp;        
	r2_old = r2;
	r2 = axpyNormCuda(-alpha, Ap, r_sloppy);

	// reliable update conditions
	rNorm = sqrt(r2);
	if (rNorm > maxrx) maxrx = rNorm;
	if (rNorm > maxrr) maxrr = rNorm;
	int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
	int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

	if (!updateR) {
	    beta = r2 / r2_old;
	    axpyZpbxCuda(alpha, p, x_sloppy, r_sloppy, beta);	
	} else {
	    
	    axpyCuda(alpha, p, x_sloppy);
      
	    if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);

	    MatPCDagMatPCCuda_st(r, fatlinkPrecise, longlinkPrecise, x, invert_param->kappa, 
			      tmp, invert_param->matpc_type);

	    r2 = xmyNormCuda(b, r);
	    if (x.precision != r_sloppy.precision) copyCuda(r_sloppy, r);
	    rNorm = sqrt(r2);

	    maxrr = rNorm;
	    rUpdate++;
	    
	    if (updateX) {
		xpyCuda(x, y);
		zeroCuda(x_sloppy);
		copyCuda(b, r);
		r0Norm = rNorm;

		maxrx = rNorm;
		xUpdate++;
	    }

	    beta = r2 / r2_old;
	    xpayCuda(r_sloppy, beta, p);
	}

	gettimeofday(&t1, NULL);


	k++;
	PRINTF("%d iterations, r2 = %e, time=%f\n", k, r2,TDIFF(t1, t0));

    }

    if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
    xpyCuda(y, x);

    perf->secs = stopwatchReadSeconds();

    if (k==invert_param->maxiter) 
	PRINTF("Exceeded maximum iterations %d\n", invert_param->maxiter);

    PRINTF("Residual updates = %d, Solution updates = %d\n", rUpdate, xUpdate);

    float gflops = (k+rUpdate)*(1.0e-9*x.volume)*(2*(2*1146+12) + 10*spinorSiteSize);
    //PRINTF("%f gflops\n", k*gflops / stopwatchReadSeconds());
    perf->gflops = gflops;
    perf->iter = k;

    //dslashCuda_st(tmp, fatlinkPrecise, longlinkPrecise, x, 1, 0);

    perf->rUpdate=rUpdate;
    perf->xUpdate=xUpdate;


#if 0
    // Calculate the true residual
    MatPCDagMatPCCuda(Ap, gauge, x, perf->kappa, tmp, perf->matpc_type);
    copyCuda(r, b);
    mxpyCuda(Ap, r);
    double true_res = normCuda(r);
    
    PRINTF("Converged after %d iterations, r2 = %e, true_r2 = %e\n", 
	   k, r2, true_res / b2);
#endif

    if (invert_param->cuda_prec_sloppy != x.precision) {
	freeParitySpinor(tmp_sloppy);
	freeParitySpinor(r_sloppy);
	freeParitySpinor(x_sloppy);
    }

    
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeParitySpinor(b);
    freeParitySpinor(r);
    freeParitySpinor(p);
    freeParitySpinor(Ap);

    freeParitySpinor(y);



    return k;
}



int 
invertCgCuda_milc_parity(ParitySpinor x, ParitySpinor source, FullGauge fatlinkPrecise, FullGauge longlinkPrecise,
			 FullGauge fatlinkSloppy, FullGauge longlinkSloppy, ParitySpinor tmp, 
			 QudaInvertParam *perf, double mass, int oddBit)
{
   
    double msq_x4 = mass*mass*4;    
    ParitySpinor p = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    ParitySpinor Ap = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    ParitySpinor y = allocateParitySpinor(x.X, x.precision);
    ParitySpinor r = allocateParitySpinor(x.X, x.precision);
    
    ParitySpinor b;
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
	b = allocateParitySpinor(x.X, x.precision);
	copyCuda(b, source);
    } else {
	b = source;
    }

    ParitySpinor x_sloppy, r_sloppy, tmp_sloppy;
    
    if (invert_param->cuda_prec_sloppy == x.precision) {
	x_sloppy = x;
	r_sloppy = r;
	tmp_sloppy = tmp;
    } else {
	x_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
	copyCuda(x_sloppy, x);
	r_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
	tmp_sloppy = allocateParitySpinor(x.X, invert_param->cuda_prec_sloppy);
    }
    
    //copyCuda(r, b);
    /* r <- b -A*x */
    dslashCuda_st(tmp, fatlinkPrecise, longlinkPrecise, x, 1 - oddBit, 0);
    dslashAxpyCuda(r, fatlinkPrecise, longlinkPrecise, tmp, oddBit, 0, x, msq_x4);    
    double r2 = xmyNormCuda(b, r);

    if (r_sloppy.precision != r.precision) copyCuda(r_sloppy, r);
    copyCuda(p, r_sloppy);
    //zeroCuda(x_sloppy);
    zeroCuda(y);

    //double b2 = 0.0;
    //b2 = normCuda(b);
    
    double r2_old;
    double src_norm = normCuda(b);
    double stop = src_norm*perf->tol*perf->tol; // stopping condition of solver
    
    double alpha, beta;
    double pAp;
    
    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = invert_param->reliable_delta;
    
    int k=1; //we already did one Dslash*Dslash operation above
    int xUpdate = 0, rUpdate = 0;
    
    PRINTF("%d iterations, r2 = %e\n", k, r2);
    stopwatchStart();
    while (r2 > stop && k<perf->maxiter) {
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	dslashCuda_st(tmp_sloppy, fatlinkSloppy, longlinkSloppy, p, 1 - oddBit, 0);
	dslashAxpyCuda(Ap, fatlinkSloppy, longlinkSloppy, tmp_sloppy, oddBit, 0, p, msq_x4);
     
	pAp = reDotProductCuda(p, Ap);

	alpha = r2 / pAp;        
	r2_old = r2;
	r2 = axpyNormCuda(-alpha, Ap, r_sloppy);

	// reliable update conditions
	rNorm = sqrt(r2);
	if (rNorm > maxrx) maxrx = rNorm;
	if (rNorm > maxrr) maxrr = rNorm;
	int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
	int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

	if (!updateR) {
	    beta = r2 / r2_old;
	    axpyZpbxCuda(alpha, p, x_sloppy, r_sloppy, beta);	
	} else {	    
	    axpyCuda(alpha, p, x_sloppy);	    
	    if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
	    dslashCuda_st(tmp, fatlinkPrecise, longlinkPrecise, x, 1 - oddBit, 0);
	    dslashAxpyCuda(r, fatlinkPrecise, longlinkPrecise, tmp, oddBit, 0, x, msq_x4);
	    
	    
	    r2 = xmyNormCuda(b, r);
	    if (x.precision != r_sloppy.precision) copyCuda(r_sloppy, r);
	    rNorm = sqrt(r2);

	    maxrr = rNorm;
	    rUpdate++;
	    k++;
 
	    if (updateX) {
		xpyCuda(x, y);
		zeroCuda(x_sloppy);
		copyCuda(b, r);
		r0Norm = rNorm;

		maxrx = rNorm;
		xUpdate++;
	    }

	    beta = r2 / r2_old;
	    xpayCuda(r_sloppy, beta, p);
	}

	gettimeofday(&t1, NULL);

	k++;
	PRINTF("%d iterations, r2 = %e, time=%f\n", k, r2,TDIFF(t1, t0));

    }

    if (x.precision != x_sloppy.precision) copyCuda(x, x_sloppy);
    xpyCuda(y, x);

    perf->secs = stopwatchReadSeconds();

    if (k==invert_param->maxiter) 
	PRINTF("Exceeded maximum iterations %d\n", invert_param->maxiter);

    PRINTF("Residual updates = %d, Solution updates = %d\n", rUpdate, xUpdate);

    float gflops = k*(1.0e-9*x.volume)*(2*1146+12 + 10*spinorSiteSize);
    perf->gflops = gflops;
    perf->iter = k;
    perf->rUpdate=rUpdate;
    perf->xUpdate=xUpdate;

#if 0
    // Calculate the true residual
    dslashCuda_st(tmp, fatlinkPrecise, longlinkPrecise, x,  1-oddBit, 0);
    dslashAxpyCuda(Ap, fatlinkPrecise, longlinkPrecise, tmp, oddBit, 0, x, msq_x4);
 
    copyCuda(r, source);
    mxpyCuda(Ap, r);
    double true_res = normCuda(r);
    
    PRINTF("Converged after %d iterations, res = %e, b2=%e, true_res = %e\n", 
           k, true_res, b2, (true_res / b2));
#endif

    if (invert_param->cuda_prec_sloppy != x.precision) {
	freeParitySpinor(tmp_sloppy);
	freeParitySpinor(r_sloppy);
	freeParitySpinor(x_sloppy);
    }
    
    
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeParitySpinor(b);
    freeParitySpinor(r);
    freeParitySpinor(p);
    freeParitySpinor(Ap);

    freeParitySpinor(y);



    return k;
}



/* 
 *   we assume the lowest mass is in offsets[0]
 */
int 
invertCgCuda_milc_multi_mass_parity(ParitySpinor* x, ParitySpinor source, FullGauge fatlinkPrecise, FullGauge longlinkPrecise,
				    FullGauge fatlinkSloppy, FullGauge longlinkSloppy, ParitySpinor tmp, QudaInvertParam *perf, 
				    double* offsets, int num_offsets, int oddBit, double* residue_sq)
{

    if (num_offsets == 0){
	return 0;
    }
    int finished[num_offsets];
    double shifts[num_offsets];
    double zeta_i[num_offsets], zeta_im1[num_offsets], zeta_ip1[num_offsets];
    double beta_i[num_offsets], beta_im1[num_offsets], alpha[num_offsets];
    int i, j;
    

    int j_low = 0;   
    int num_offsets_now = num_offsets;
    for(i=0;i <num_offsets;i++){
	finished[i]= 0;
	shifts[i] = offsets[i] - offsets[0];
	zeta_im1[i] = zeta_i[i] = 1.0;
	beta_im1[i] = -1.0;
	alpha[i] =0.0;
    }

    int* dims = source.X;
    
    double msq_x4 = offsets[0];
    
    ParitySpinor p[num_offsets];
    ParitySpinor y[num_offsets];

    for(i=0;i < num_offsets;i++){
	p[i]= allocateParitySpinor(dims, invert_param->cuda_prec_sloppy);
	y[i] = allocateParitySpinor(dims, source.precision);
    }
    
    ParitySpinor Ap = allocateParitySpinor(dims, invert_param->cuda_prec_sloppy);
    ParitySpinor r = allocateParitySpinor(dims, source.precision);
    
    ParitySpinor b;
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) {
	b = allocateParitySpinor(dims, source.precision);
	copyCuda(b, source);
    } else {
	b = source;
    }
    
    ParitySpinor x_sloppy[num_offsets], r_sloppy, tmp_sloppy;
    
    if (invert_param->cuda_prec_sloppy == source.precision) {
	for(i=0;i < num_offsets;i++){
	    x_sloppy[i] = x[i];
	}
	r_sloppy = r;
	tmp_sloppy = tmp;	
    } else {
	for(i=0;i < num_offsets;i++){
	    x_sloppy[i] = allocateParitySpinor(dims, invert_param->cuda_prec_sloppy);
	}
	r_sloppy = allocateParitySpinor(dims, invert_param->cuda_prec_sloppy);
	tmp_sloppy = allocateParitySpinor(dims, invert_param->cuda_prec_sloppy);
    }
    
    copyCuda(r, b);
    if (r_sloppy.precision != r.precision) copyCuda(r_sloppy, r);

    for(i=0;i < num_offsets; i++){
	copyCuda(p[i], r_sloppy);
	zeroCuda(x_sloppy[i]);
	zeroCuda(y[i]);
    }
    
    double b2 = 0.0;
    b2 = normCuda(b);
    
    double r2 = b2;
    double r2_old;
    double stop = r2*perf->tol*perf->tol; // stopping condition of solver
    
    double pAp;
    
    int k=0;
    int xUpdate = 0, rUpdate = 0;
    
    PRINTF("%d iterations, r2 = %e\n", k, r2);
    stopwatchStart();
    while (r2 > stop && k<perf->maxiter) {
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	//MatDagMatCuda(Ap, fatlinkSloppy, longlinkSloppy, p, perf->kappa, tmp_sloppy, perf->matpc_type);
	
	//MatPCDagMatPCCuda_st(Ap, fatlinkSloppy, longlinkSloppy, p, perf->kappa, tmp_sloppy, perf->matpc_type);
	dslashCuda_st(tmp_sloppy, fatlinkSloppy, longlinkSloppy, p[0], 1 - oddBit, 0);
	dslashAxpyCuda(Ap, fatlinkSloppy, longlinkSloppy, tmp_sloppy, oddBit, 0, p[0], msq_x4);

	pAp = reDotProductCuda(p[0], Ap);
	beta_i[0] = r2 / pAp;        

	zeta_ip1[0] = 1.0;
	for(j=1;j<num_offsets_now;j++) {
	    zeta_ip1[j] = zeta_i[j] * zeta_im1[j] * beta_im1[j_low];
	    double c1 = beta_i[j_low] * alpha[j_low] * (zeta_im1[j]-zeta_i[j]);
	    double c2 = zeta_im1[j] * beta_im1[j_low] * (1.0+shifts[j]*beta_i[j_low]);
	    /*THISBLOWSUP
	      zeta_ip1[j] /= c1 + c2;
	      beta_i[j] = beta_i[j_low] * zeta_ip1[j] / zeta_i[j];
	    */
	    /*TRYTHIS*/
            if( (c1+c2) != 0.0 )
		zeta_ip1[j] /= (c1 + c2); 
	    else {
		zeta_ip1[j] = 0.0;
		finished[j] = 1;
	    }
            if( zeta_i[j] != 0.0){
                beta_i[j] = beta_i[j_low] * zeta_ip1[j] / zeta_i[j];
            } else  {
		zeta_ip1[j] = 0.0;
		beta_i[j] = 0.0;
		finished[j] = 1;
		PRINTF("SETTING A ZERO, j=%d, num_offsets_now=%d\n",j,num_offsets_now);
		//if(j==num_offsets_now-1)node0_PRINTF("REDUCING OFFSETS\n");
		if(j==num_offsets_now-1)num_offsets_now--;
		// don't work any more on finished solutions
		// this only works if largest offsets are last, otherwise
		// just wastes time multiplying by zero
            }
	}	
	


	r2_old = r2;

	r2 = axpyNormCuda(-beta_i[j_low], Ap, r_sloppy);

	    
	alpha[0] = r2 / r2_old;
	
	for(j=1;j<num_offsets_now;j++){
	    /*THISBLOWSUP
	      alpha[j] = alpha[j_low] * zeta_ip1[j] * beta_i[j] /
	      (zeta_i[j] * beta_i[j_low]);
	    */
	    /*TRYTHIS*/
	    if( zeta_i[j] * beta_i[j_low] != 0.0)
		alpha[j] = alpha[j_low] * zeta_ip1[j] * beta_i[j] /
		    (zeta_i[j] * beta_i[j_low]);
	    else {
		alpha[j] = 0.0;
		finished[j] = 1;
	    }
	}
	
	
	axpyZpbxCuda(beta_i[0], p[0], x_sloppy[0], r_sloppy, alpha[0]);	
	for(j=1;j < num_offsets_now; j++){
	    axpyBzpcxCuda(beta_i[j], p[j], x_sloppy[j], zeta_ip1[j], r_sloppy, alpha[j]);
	}
	
	gettimeofday(&t1, NULL);
	
	for(j=0;j<num_offsets_now;j++){
	    beta_im1[j] = beta_i[j];
	    zeta_im1[j] = zeta_i[j];
	    zeta_i[j] = zeta_ip1[j];
	}
	
	
	k++;
	PRINTF("%d iterations, r2 = %e, time=%f\n", k, r2,TDIFF(t1, t0));

    }
    
    if (x[0].precision != x_sloppy[0].precision) {
	for(i=0;i < num_offsets; i++){
	    copyCuda(x[i], x_sloppy[i]);
	}
    }

    for(i=0;i < num_offsets;i++){
	xpyCuda(y[i], x[i]);
    }
    
    *residue_sq = r2;
    

    perf->secs = stopwatchReadSeconds();
    
    if (k==invert_param->maxiter) 
	PRINTF("Exceeded maximum iterations %d\n", invert_param->maxiter);
    
    PRINTF("Residual updates = %d, Solution updates = %d\n", rUpdate, xUpdate);
    
    float gflops = k*(1.0e-9*x[0].volume)*(2*1146+12 + 10*spinorSiteSize);
    //PRINTF("%f gflops\n", k*gflops / stopwatchReadSeconds());
    perf->gflops = gflops;
    perf->iter = k;
    perf->rUpdate=rUpdate;
    perf->xUpdate=xUpdate;
#if 0
    // Calculate the true residual
    dslashCuda_st(tmp, fatlinkPrecise, longlinkPrecise, x[0],  1-oddBit, 0);
    dslashAxpyCuda(Ap, fatlinkPrecise, longlinkPrecise, tmp, oddBit, 0, x[0], msq_x4);
 
    copyCuda(r, source);
    mxpyCuda(Ap, r);
    double true_res = normCuda(r);
    
    PRINTF("Converged after %d iterations, res = %e, b2=%e, true_res = %e\n", 
           k, true_res, b2, (true_res / b2));
#endif
    
    if (invert_param->cuda_prec_sloppy != x[0].precision) {
	freeParitySpinor(tmp_sloppy);
	freeParitySpinor(r_sloppy);
	for(i=0;i < num_offsets;i++){
	    freeParitySpinor(x_sloppy[i]);
	}
    }
    
    
    if (perf->preserve_source == QUDA_PRESERVE_SOURCE_YES) freeParitySpinor(b);
    freeParitySpinor(r);
    
    for(i=0;i < num_offsets; i++){
	freeParitySpinor(p[i]);
	freeParitySpinor(y[i]);
    }
    freeParitySpinor(Ap);



    return k;
}


