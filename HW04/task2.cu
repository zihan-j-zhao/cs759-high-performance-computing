#include <cuda.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "stencil.cuh"

// Reference: FAQ/BestPractices/random_numbers.md
void generate(float *output, std::size_t n, float lower, float upper) {
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(lower, upper);
  for (std::size_t i = 0; i < n; ++i) {
      output[i] = dist(generator);
  }
}
// ===== above =====

int main(int argc, char *argv[]) {
  const unsigned int n = atol(argv[1]);
  const unsigned int R = atol(argv[2]);
  const unsigned int threads_per_block = atol(argv[3]);
  const unsigned int MASK_SIZE = 2 * R + 1;

  float *dImage, *dMask, *dOutput, *hImage, *hMask, *hOutput;
  hImage = new float[n];
  hMask = new float[MASK_SIZE];
  hOutput = new float[n];
  cudaMalloc((void **)&dImage, n * sizeof(float));
  cudaMalloc((void **)&dMask, MASK_SIZE * sizeof(float));
  cudaMalloc((void **)&dOutput, n * sizeof(float));
  
  // Generate random numbers
  generate(hImage, n, -1.0, 1.0);
  generate(hMask, MASK_SIZE, -1.0, 1.0);
  cudaMemcpy(dImage, hImage, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dMask, hMask, MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  stencil(dImage, dMask, dOutput, n, R, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  cudaMemcpy(hOutput, dOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << hOutput[n - 1] << std::endl;
  std::cout << ms << std::endl;
  
  delete[] hImage;
  delete[] hMask;
  delete[] hOutput;
  cudaFree(dImage);
  cudaFree(dMask);
  cudaFree(dOutput);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
