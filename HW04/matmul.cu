#include "matmul.cuh"

__global__ void matmul_kernel(const float *A, const float *B, size_t *C, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * n) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      const int row = idx / n;
      const int col = idx % n;
      sum += A[row * n + i] * B[col + i * n];
    }
    C[idx] = sum;
  }
}

void matmul(const float *A, const float *B, float *C, size_t n, unsigned int threads_per_block) {
  const unsigned int num_blocks = (n * n + threads_per_block - 1) / threads_per_block;
  matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
  cudaDeviceSynchronize();
}
