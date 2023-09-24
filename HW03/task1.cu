#include <cuda.h>
#include <iostream>

__global__ void fact() {
  int num = threadIdx.x + 1;
  int res = 1;
  for (int i = 1; i <= num; ++i) {
    res *= i;
  }
  std::printf("%d!=%d\n", num, res);
}

int main() {
  fact<<<1, 8>>>();
  cudaDeviceSynchronize();
  return 0;
}
