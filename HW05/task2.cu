#include <cuda.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "reduce.cuh"

// Reference: FAQ/BestPractices/random_numbers.md
template<typename T>
void generate(T *output, std::size_t n, float lower, float upper) {
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(lower, upper);
  for (std::size_t i = 0; i < n; ++i) {
    output[i] = (T) dist(generator);
  }
}
// ===== above =====

int main(int argc, char *argv[]) {
  const size_t n = atol(argv[1]);
  const size_t threads_per_block = atol(argv[2]);
  const size_t num_blocks = (n + 2 * threads_per_block - 1) / (2 * threads_per_block);

  float *dA, *dB, *hA, *hB;
  hA = new float[n];
  hB = new float[num_blocks];
  cudaMalloc((void **)&dA, n * sizeof(float));
  cudaMalloc((void **)&dB, num_blocks * sizeof(float));

  // Generate random numbers
  generate<float>(hA, n, -1.0, 1.0);
  cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  reduce((float **)dA, (float **)dB, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  cudaMemcpy(hB, dB, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << hB[0] << std::endl << ms << std::endl;

  delete[] hA; delete[] hB;
  cudaFree(dA); cudaFree(dB);

  return 0;
}
