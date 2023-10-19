#include <cuda.h>
#include <cublas_v2.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "mmul.h"


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
  const int n = atoi(argv[1]);
  const int n_tests = atoi(argv[2]);

  float *A, *B, *C;
  cudaMallocManaged((void **)&A, n * n * sizeof(float));
  cudaMallocManaged((void **)&B, n * n * sizeof(float));
  cudaMallocManaged((void **)&C, n * n * sizeof(float));

  generate<float>(A, n * n, -1.0, 1.0);
  generate<float>(B, n * n, -1.0, 1.0);
  generate<float>(C, n * n, -1.0, 1.0);

  cublasHandle_t handle;
	cublasCreate(&handle);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < n_tests; ++i) {
    mmul(handle, A, B, C, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  std::cout << ms / (float) n_tests << std::endl;

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
