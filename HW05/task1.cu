#include <cuda.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "matmul.cuh"

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

template<typename T>
void call_func(void (*f) (const T*, const T*, T*, unsigned int, unsigned int), unsigned int n, unsigned int block_dim) {
  // Initialize variables
  size_t size = n * n * sizeof(T);
  T *dA, *dB, *dC, *hA, *hB, *hC;
  hA = new T[n * n];
  hB = new T[n * n];
  hC = new T[n * n];
  cudaMalloc((void **)&dA, size);
  cudaMalloc((void **)&dB, size);
  cudaMalloc((void **)&dC, size);

  generate<T>(hA, n * n, -10.0, 10.0);
  generate<T>(hB, n * n, -10.0, 10.0);
  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  f(dA, dB, dC, n, block_dim);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

  std::cout << hC[0] << std::endl << hC[n * n - 1] << std::endl;
  std::cout << ms << std::endl;
  
  delete[] hA;
  delete[] hB;
  delete[] hC;
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
  const size_t n = atol(argv[1]);
  const size_t block_dim = atol(argv[2]);

  call_func<int>(matmul_1, n, block_dim);
  call_func<float>(matmul_2, n, block_dim);
  call_func<double>(matmul_3, n, block_dim);

  return 0;
}
