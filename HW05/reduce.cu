#include "reduce.cuh"

// Reference: Lecture 14 slides (modified)
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + tid;
  sdata[tid] = i + blockDim.x < n ? g_idata[i] + g_idata[i + blockDim.x] : i < n ? g_idata[i] : 0;
  __syncthreads();
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}
// ===== above =====

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
  const unsigned int sh_size = threads_per_block * sizeof(float);
  unsigned int i = N;
  while (i > 1) {
    const unsigned int num_blocks = (i + 2 * threads_per_block - 1) / (2 * threads_per_block);
    reduce_kernel<<<num_blocks, threads_per_block, sh_size>>>((float *)input, (float *)output, i);
    cudaMemcpy(input, output, num_blocks * sizeof(float), cudaMemcpyDeviceToDevice);
    i = num_blocks;
  }
  cudaDeviceSynchronize();
}
