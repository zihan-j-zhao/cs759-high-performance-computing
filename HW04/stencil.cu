#include "stencil.cuh"

__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R) {
  const int tx = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tx;
  if (idx >= n) {  // do nothing if idx out of bound
    return;
  }

  // Each block's shared memory has an entire copy of mask, a copy of a part of image, and a copy of a part of output.
  // The parts of image and output have the same size equal to the number of threads in the block. However, in the
  // shared memory, the part of image additionally has a front and a rear R * sizeof(float) bytes of padding to deal 
  // with the edge cases when index of image may be out of bound.

  // set up shared memory in each block
  extern __shared__ float shared[];  // MEMLAYOUT => (MASK_SIZE) + (IMAGE_PAD + BUFF_SIZE + IMAGE_PAD) + (BUFF_SIZE)
  const int _R = (int)R;
  const unsigned int IMAGE_PAD = R;
  const unsigned int MASK_SIZE = 2 * R + 1;
  const unsigned int BUFF_SIZE = blockDim.x;

  float *shared_mask = shared;
  float *shared_image = shared_mask + MASK_SIZE + IMAGE_PAD;
  float *shared_output = shared_image + BUFF_SIZE + IMAGE_PAD;

  // store mask in shared memory of _each block in parallel_
  if (tx < MASK_SIZE) {
    shared_mask[tx] = mask[tx];
  }

  // store image in shared memory of _each block in parallel_
  shared_image[tx] = image[idx];

  // initialize front padded entries to 1 if index of image < 0
  if (tx < _R) {
    shared_image[tx - _R] = idx - _R > 0 ? image[idx - _R] : 1;
  }

  // initialize rear padded entries to 1 if index of image >= n
  else if (tx + _R >= BUFF_SIZE) {
    shared_image[tx + _R] = idx + _R < n ? image[idx + _R] : 1;
  }

  // initialize output entries in shared memory of _each block in parallel_
  shared_output[tx] = 0;
  

  __syncthreads();


  // compute using shared memory
  for (int j = -_R; j <= _R; ++j) {
    // without shared memory: output[i] = image[i + j] * mask[j + R];
    shared_output[tx] += shared_image[tx + j] * shared_mask[j + _R];
  }

  output[idx] = shared_output[tx];
}

void stencil(const float *image, const float *mask, float *output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
  const int num_blocks = (threads_per_block + n - 1) / threads_per_block;
  const int shared_size = (2 * R + 1 + R + threads_per_block + R + threads_per_block) * sizeof(float);
  stencil_kernel<<<num_blocks, threads_per_block, shared_size>>>(image, mask, output, n, R);
  cudaDeviceSynchronize();
}
