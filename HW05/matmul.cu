#include "matmul.cuh"

#include <iostream>


// Reference: Lecture 11 slides (modified)
template<typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
  int BLOCK_SIZE = blockDim.x; // same as blockDim.y in this case

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int aBegin = n * BLOCK_SIZE * by;
  int aEnd = aBegin + n - 1;
  int aStep = BLOCK_SIZE;
  
  int bBegin = BLOCK_SIZE * bx;
  int bStep = BLOCK_SIZE * n;

  // Templated initialization
  // Reference: https://stackoverflow.com/a/27570775/11247758
  extern __shared__ char shmem[];
  T *shared = reinterpret_cast<T*>(shmem);
  // ===== above =====

  T *Asub = shared;
  T *Bsub = Asub + BLOCK_SIZE * BLOCK_SIZE;
  T Csub = 0;

  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    int idx = ty * BLOCK_SIZE + tx;
    int idx_a = a + n * ty + tx;
    int idx_b = b + n * ty + tx;

    Asub[idx] = idx_a < n * n ? A[idx_a] : 0;
    Bsub[idx] = idx_b < n * n ? B[idx_b] : 0;

    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j) {
      Csub += Asub[ty * BLOCK_SIZE + j] * Bsub[j * BLOCK_SIZE + tx];
    }

    __syncthreads();
  }

  int c = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  int idx = c + n * ty + tx;
  if (idx < n * n) {
    C[idx] = Csub;
  }
}
// ===== above =====

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
  unsigned int grid_dim = (n + block_dim - 1) / block_dim;

  dim3 dimBlock(block_dim, block_dim);
  dim3 dimGrid(grid_dim, grid_dim);
  unsigned int shSize = 2 * block_dim * block_dim * sizeof(int);

  matmul_kernel<int><<<dimGrid, dimBlock, shSize>>>(A, B, C, n);

  cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
  unsigned int grid_dim = (n + block_dim - 1) / block_dim;

  dim3 dimBlock(block_dim, block_dim);
  dim3 dimGrid(grid_dim, grid_dim);
  unsigned int shSize = 2 * block_dim * block_dim * sizeof(float);

  matmul_kernel<float><<<dimGrid, dimBlock, shSize>>>(A, B, C, n);

  cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
  unsigned int grid_dim = (n + block_dim - 1) / block_dim;

  dim3 dimBlock(block_dim, block_dim);
  dim3 dimGrid(grid_dim, grid_dim);
  unsigned int shSize = 2 * block_dim * block_dim * sizeof(double);

  matmul_kernel<double><<<dimGrid, dimBlock, shSize>>>(A, B, C, n);

  cudaDeviceSynchronize();
}

