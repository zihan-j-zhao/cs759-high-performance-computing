// Reference: GPU/CUB-related/deviceReduce.cu
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include <cmath>
#include <random>
#include <iostream>


using namespace cub;
CachingDeviceAllocator g_allocator(true);


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

  float *h_in, *d_in, *d_sum, gpu_sum;
  h_in = new float[n];
  generate<float>(h_in, n, -1.0, 1.0);

  g_allocator.DeviceAllocate((void **)&d_in, sizeof(float) * n);
  g_allocator.DeviceAllocate((void **)&d_sum, sizeof(float));
  cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice);

  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
  g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
  
  // Reference: Assignments/general/timing.md
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  // ===== above =====

  cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << gpu_sum << std::endl << ms << std::endl;

  delete[] h_in;
  g_allocator.DeviceFree(d_in);
  g_allocator.DeviceFree(d_sum);
  g_allocator.DeviceFree(d_temp_storage);

  return 0;
}
// ===== above ====
