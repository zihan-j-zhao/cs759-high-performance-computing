#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <random>
#include <iostream>


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
  const unsigned int n = atoi(argv[1]);

  float *data = new float[n];
  generate<float>(data, n, -1.0, 1.0);
  thrust::host_vector<float> hvec(n);
  for (unsigned int i = 0; i < n; ++i) hvec[i] = data[i];
  thrust::device_vector<float> dvec = hvec;
  
  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  float res = thrust::reduce(dvec.begin(), dvec.end(), 0.0, thrust::plus<float>());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  std::cout << res << std::endl << ms << std::endl;

  delete[] data;

  return 0;
}
