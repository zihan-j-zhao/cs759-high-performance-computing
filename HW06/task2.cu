#include <cuda.h>

#include <cmath>
#include <random>
#include <cstring>
#include <iostream>

#include "scan.cuh"


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
  const int threads_per_block = atoi(argv[2]);

  float *input, *output;
  cudaMallocManaged((void**)&input, n * sizeof(float));
  cudaMallocManaged((void**)&output, n * sizeof(float));

  generate<float>(input, n, -1.0, 1.0);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  scan(input, output, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  std::cout << output[n - 1] << std::endl << ms << std::endl;

  cudaFree(input);
  cudaFree(output);

  return 0;
}
