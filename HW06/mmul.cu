#include "mmul.h"

void mmul(cublasHandle_t handle, const float *A, const float *B, float *C, int n) {
  float alpha = 1.0;
  float beta = 1.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
  cudaDeviceSynchronize();
}
