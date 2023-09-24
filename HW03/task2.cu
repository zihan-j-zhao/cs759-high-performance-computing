#include <cuda.h>
#include <random>
#include <iostream>

__global__ void f(int *arr, int a) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  arr[blockIdx.x * blockDim.x + threadIdx.x] = a * x + y;
}

int main() {
  const int N_ELE = 16;
  int *dA, hA[N_ELE], a;

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_int_distribution<> dist(0, 10);

  a = dist(generator);

  cudaMalloc((void **)&dA, N_ELE * sizeof(int));

  f<<<2, 8>>>(dA, a);
  cudaDeviceSynchronize();
  
  cudaMemcpy(hA, dA, N_ELE * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dA);

  for (int i = 0; i < N_ELE; ++i) {
    if (i + 1 == N_ELE) {
      std::cout << hA[i] << std::endl;
    } else {
      std::cout << hA[i] << " ";
    }
  }

  return 0;
}
