#include "scan.cuh"

#include <iostream>

// Reference: Lecture 14 slides (modified)
__global__ void hillis_steele(float *g_odata, float *g_idata, int n, float *buf) {
  extern volatile __shared__ float temp[];

  int thid = threadIdx.x;
  int dim = blockDim.x;
  int idx = blockIdx.x * blockDim.x + thid;
  int pout = 0, pin = 1;
  
  if (idx >= n) {
    return;
  }

  temp[thid] = g_idata[idx];
  __syncthreads();

  for (int offset = 1; offset < dim; offset *= 2) {
    pout = 1 - pout;
    pin = 1 - pout;

    temp[pout * dim + thid] = temp[pin * dim + thid];
    if (thid >= offset) {
      temp[pout * dim + thid] += temp[pin * dim + thid - offset];
    }

    __syncthreads();
  }

  if (pout * dim + thid < dim) {
    g_odata[idx] = temp[pout * n + thid];
  }

  // parallel move to temp results
  if (buf != NULL && thid == dim - 1) {
    buf[blockIdx.x] = temp[pout * n + thid];
  }
}
// ===== above =====

// parallel add
__global__ void padd(float *a, float *b, float *c, unsigned int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  //c[idx] = a[idx];
  if (blockIdx.x != 0) {
    a[idx] += b[blockIdx.x - 1];
  }
}

__host__ void scan(const float *input, float *output, unsigned int n, unsigned int threads_per_block) {
  const unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  const unsigned int sh_size = 2 * threads_per_block * sizeof(float);

  // move data from managed memory to device memory for manipulations
  float *g_idata, *g_odata, *buffer, *g_osec;
  cudaMallocManaged((void**)&g_idata, n * sizeof(float));
  cudaMallocManaged((void**)&g_odata, n * sizeof(float));
  cudaMallocManaged((void**)&buffer, num_blocks * sizeof(float));
  cudaMallocManaged((void**)&g_osec, num_blocks * sizeof(float));

  cudaMemcpy(g_idata, input, n * sizeof(float), cudaMemcpyHostToDevice);
  
  // call kernel
  hillis_steele<<<num_blocks, threads_per_block, sh_size>>>(g_odata, g_idata, n, buffer);
  hillis_steele<<<1, threads_per_block, sh_size>>>(g_osec, buffer, num_blocks, NULL);
  padd<<<num_blocks, threads_per_block>>>(g_odata, g_osec, output, n);

  cudaDeviceSynchronize();
  cudaMemcpy(output, g_odata, n * sizeof(float), cudaMemcpyDeviceToHost);

  // clean
  cudaFree(g_idata);
  cudaFree(g_odata);
  cudaFree(buffer);
  cudaFree(g_osec);
}

