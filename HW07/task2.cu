#include <cuda.h>

#include <cmath>
#include <random>
#include <iostream>

#include "count.cuh"


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
  const unsigned int n = atol(argv[1]);

  int *data = new float[n];
  generate<int>(data, n, 0, 500);
  thrust::host_vector<int> hvec(n);
  for (unsigned int i = 0; i < n; ++i) hvec[i] = data[i];
  thrust::device_vector<int> dvec = hvec;
  thrust::device_vector<int> values(n);
  thrust::device_vector<int> counts(n);

  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  count(dvec, values, counts);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  std::cout << values.back() << std::endl << counts.back() << std::endl << ms << std::endl;

  delete[] data;

  return 0;
}
