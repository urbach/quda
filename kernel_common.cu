
__constant__ int X1h;
__constant__ int X1;
__constant__ int X2;
__constant__ int X3;
__constant__ int X4;

__constant__ int X1m1;
__constant__ int X2m1;
__constant__ int X3m1;
__constant__ int X4m1;

__constant__ int X2X1mX1;
__constant__ int X3X2X1mX2X1;
__constant__ int X4X3X2X1mX3X2X1;
__constant__ int X4X3X2X1hmX3X2X1h;

__constant__ float X1h_inv;
__constant__ float X2_inv;
__constant__ float X3_inv;

__constant__ int X2X1;
__constant__ int X3X2X1;

__constant__ int Vh;
__constant__ int V;

__constant__ int Vhx2;
__constant__ int Vhx3;
__constant__ int Vhx4;
__constant__ int Vhx5;
__constant__ int Vhx6;
__constant__ int Vhx7;
__constant__ int Vhx8;
__constant__ int Vhx9;


static int init_kernel_cuda_flag = 0;
void
init_kernel_cuda(QudaGaugeParam* param)
{
    if (init_kernel_cuda_flag){
	return;
    }
    init_kernel_cuda_flag =1;
    
    int X1 = param->X[0];
    cudaMemcpyToSymbol("X1", &X1, sizeof(int));  
    
    int X2 = param->X[1];
    cudaMemcpyToSymbol("X2", &X2, sizeof(int));  
    
    int X3 = param->X[2];
    cudaMemcpyToSymbol("X3", &X3, sizeof(int));  
    
    int X4 = param->X[3];
    cudaMemcpyToSymbol("X4", &X4, sizeof(int));  
    
    int X2X1 = X2*X1;
    cudaMemcpyToSymbol("X2X1", &X2X1, sizeof(int));  
    
    int X3X2X1 = X3*X2*X1;
    cudaMemcpyToSymbol("X3X2X1", &X3X2X1, sizeof(int));  
    
    int X1h = X1/2;
    cudaMemcpyToSymbol("X1h", &X1h, sizeof(int));  
    
    float X1h_inv = 1.0 / X1h;
    cudaMemcpyToSymbol("X1h_inv", &X1h_inv, sizeof(float));  
    
    float X2_inv = 1.0 / X2;
    cudaMemcpyToSymbol("X2_inv", &X2_inv, sizeof(float));  
    
    float X3_inv = 1.0 / X3;
    cudaMemcpyToSymbol("X3_inv", &X3_inv, sizeof(float));  
    
    int X1m1 = X1 - 1;
    cudaMemcpyToSymbol("X1m1", &X1m1, sizeof(int));  

    int X2m1 = X2 - 1;
    cudaMemcpyToSymbol("X2m1", &X2m1, sizeof(int));  
    
    int X3m1 = X3 - 1;
    cudaMemcpyToSymbol("X3m1", &X3m1, sizeof(int));  
    
    int X4m1 = X4 - 1;
    cudaMemcpyToSymbol("X4m1", &X4m1, sizeof(int));  
    
    int X2X1mX1 = X2X1 - X1;
    cudaMemcpyToSymbol("X2X1mX1", &X2X1mX1, sizeof(int));  
    
    int X3X2X1mX2X1 = X3X2X1 - X2X1;
    cudaMemcpyToSymbol("X3X2X1mX2X1", &X3X2X1mX2X1, sizeof(int));  
    
    int X4X3X2X1mX3X2X1 = (X4-1)*X3X2X1;
    cudaMemcpyToSymbol("X4X3X2X1mX3X2X1", &X4X3X2X1mX3X2X1, sizeof(int));  
    
    int X4X3X2X1hmX3X2X1h = (X4-1)*X3*X2*X1h;
    cudaMemcpyToSymbol("X4X3X2X1hmX3X2X1h", &X4X3X2X1hmX3X2X1h, sizeof(int));  

    int Vh = param->X[0]*param->X[1]*param->X[2]*param->X[3]/2;
    int V = 2*Vh;
    int Vhx2 = 2*Vh;
    int Vhx3 = 3*Vh;
    int Vhx4 = 4*Vh;
    int Vhx5 = 5*Vh;
    int Vhx6 = 6*Vh;
    int Vhx7 = 7*Vh;
    int Vhx8 = 8*Vh;
    int Vhx9 = 9*Vh;

    cudaMemcpyToSymbol("Vh", &Vh, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("V", &V, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx2", &Vhx2, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx3", &Vhx3, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx4", &Vhx4, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx5", &Vhx5, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx6", &Vhx6, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx7", &Vhx7, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx8", &Vhx8, sizeof(int)); CUERR;
    cudaMemcpyToSymbol("Vhx9", &Vhx9, sizeof(int)); CUERR;

    return;
}
