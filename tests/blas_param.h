//
// Auto-tuned blas CUDA parameters, generated by blas_test
//

static int blas_threads[30][3] = {
  { 128,  480,  416},  // Kernel  0: copyCuda (high source precision)
  { 576,   32,  384},  // Kernel  1: copyCuda (low source precision)
  {  96,  128,  128},  // Kernel  2: axpbyCuda
  {  96,  128,  128},  // Kernel  3: xpyCuda
  { 160,  128,  128},  // Kernel  4: axpyCuda
  {  96,  128,  128},  // Kernel  5: xpayCuda
  { 160,  128,  128},  // Kernel  6: mxpyCuda
  {  96,  480,  768},  // Kernel  7: axCuda
  {  96,  128,   96},  // Kernel  8: caxpyCuda
  {  96,  128,   64},  // Kernel  9: caxpbyCuda
  { 160,   96,   96},  // Kernel 10: cxpaypbzCuda
  { 512,   64,   64},  // Kernel 11: axpyBzpcxCuda
  { 512,   64,   64},  // Kernel 12: axpyZpbxCuda
  { 128,   96,   64},  // Kernel 13: caxpbypzYmbwCuda
  { 128,  256,  256},  // Kernel 14: normCuda
  { 128,  128,  256},  // Kernel 15: reDotProductCuda
  { 256,  256,  512},  // Kernel 16: axpyNormCuda
  { 256,  256,  512},  // Kernel 17: xmyNormCuda
  { 128,  128,  256},  // Kernel 18: cDotProductCuda
  { 256,  128,  256},  // Kernel 19: xpaycDotzyCuda
  { 128,  128,  256},  // Kernel 20: cDotProductNormACuda
  { 128,  256,  256},  // Kernel 21: cDotProductNormBCuda
  { 256,  256,  256},  // Kernel 22: caxpbypzYmbwcDotProductWYNormYCuda
  { 128,  128,   64},  // Kernel 23: cabxpyAxCuda
  { 256,  256,  256},  // Kernel 24: caxpyNormCuda
  { 256,  256,  256},  // Kernel 25: caxpyXmazNormXCuda
  { 256,  512,  256},  // Kernel 26: cabxpyAxNormCuda
  { 128,  128,   64},  // Kernel 27: caxpbypzCuda
  {  64,  128,  128},  // Kernel 28: caxpbypczpwCuda
  { 256,  256,  256}   // Kernel 29: caxpyDotzyCuda
};

static int blas_blocks[30][3] = {
  { 2048, 32768,  2048},  // Kernel  0: copyCuda (high source precision)
  { 8192, 65536,  1024},  // Kernel  1: copyCuda (low source precision)
  { 2048, 16384, 32768},  // Kernel  2: axpbyCuda
  { 2048, 16384, 65536},  // Kernel  3: xpyCuda
  { 1024, 16384, 65536},  // Kernel  4: axpyCuda
  { 2048, 16384, 32768},  // Kernel  5: xpayCuda
  { 1024, 16384, 32768},  // Kernel  6: mxpyCuda
  { 2048,  4096,  8192},  // Kernel  7: axCuda
  { 2048, 65536, 65536},  // Kernel  8: caxpyCuda
  { 2048, 32768, 32768},  // Kernel  9: caxpbyCuda
  { 1024, 32768, 65536},  // Kernel 10: cxpaypbzCuda
  {  512, 32768, 32768},  // Kernel 11: axpyBzpcxCuda
  {  512, 32768, 32768},  // Kernel 12: axpyZpbxCuda
  { 2048, 32768, 65536},  // Kernel 13: caxpbypzYmbwCuda
  {   64,    64,  1024},  // Kernel 14: normCuda
  {  512,   512,  1024},  // Kernel 15: reDotProductCuda
  { 1024,    64,  4096},  // Kernel 16: axpyNormCuda
  {32768,    64,  4096},  // Kernel 17: xmyNormCuda
  {  256,   512,   512},  // Kernel 18: cDotProductCuda
  {  256,    64,  2048},  // Kernel 19: xpaycDotzyCuda
  {  128,    64,   512},  // Kernel 20: cDotProductNormACuda
  {  256,   256,   512},  // Kernel 21: cDotProductNormBCuda
  {  256,   512,   512},  // Kernel 22: caxpbypzYmbwcDotProductWYNormYCuda
  { 2048, 32768, 65536},  // Kernel 23: cabxpyAxCuda
  {  512,  1024,  1024},  // Kernel 24: caxpyNormCuda
  { 4096,  2048,  4096},  // Kernel 25: caxpyXmazNormXCuda
  { 4096,  2048,  4096},  // Kernel 26: cabxpyAxNormCuda
  { 2048, 32768, 32768},  // Kernel 27: caxpbypzCuda
  { 4096, 16384, 65536},  // Kernel 28: caxpbypczpwCuda
  {  256,  1024,  1024}   // Kernel 29: caxpyDotzyCuda
};