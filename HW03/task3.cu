#include <cuda.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "vscale.cuh"


// Reference: directly copied from my HW01/task6.cpp for numeric validation
bool is_numeric(const char *str) {
  size_t size = std::strlen(str);
  for (size_t i = 0; i < size; ++i) {
    if (!std::isdigit(str[i])) {
      return false;
    }
  }
  return true;
}
// ===== above =====


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
  if (argc != 2) {
    std::cout << "Usage: task3 n" << std::endl;
    return 1;
  }

  if (!is_numeric(argv[1])) {
    std::cout << "Usage: task3 n" << std::endl;
    return 1;
  }

  unsigned int n = std::atoi(argv[1]);
  float *dA, *dB, *hA, *hB;
  hA = new float[n];
  hB = new float[n];
  cudaMalloc((void **)&dA, n * sizeof(float));
  cudaMalloc((void **)&dB, n * sizeof(float));
  
  // Generate random numbers
  generate(hA, n, -10.0, 10.0);
  generate(hB, n, 0.0, 1.0);
  cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 16;
  int blocks = (n + threads - 1) / threads;

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  vscale<<<blocks, threads>>>(dA, dB, n);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << ms << std::endl;
  std::cout << hB[0] << std::endl << hB[n - 1] << std::endl;

  delete[] hA;
  delete[] hB;
  cudaFree(dA);
  cudaFree(dB);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
