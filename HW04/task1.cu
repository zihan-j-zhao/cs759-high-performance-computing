#include <cuda.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "matmul.cuh"

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
  const size_t n = atol(argv[1]);
  const unsigned int threads_per_block = atol(argv[2]);

  float *dA, *dB, *dC, *hA, *hB, *hC;
  hA = new float[n * n];
  hB = new float[n * n];
  hC = new float[n * n];
  cudaMalloc((void **)&dA, n * n * sizeof(float));
  cudaMalloc((void **)&dB, n * n * sizeof(float));
  cudaMalloc((void **)&dC, n * n * sizeof(float));
  
  // Generate random numbers
  generate(hA, n * n, -1.0, 1.0);
  generate(hB, n * n, -1.0, 1.0);
  cudaMemcpy(dA, hA, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, n * n * sizeof(float), cudaMemcpyHostToDevice);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  matmul(dA, dB, dC, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  cudaMemcpy(hC, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << hC[n * n - 1] << std::endl;
  std::cout << ms << std::endl;

  delete[] hA;
  delete[] hB;
  delete[] hC;
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
