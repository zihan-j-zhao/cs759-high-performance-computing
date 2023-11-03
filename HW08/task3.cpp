#include <cmath>
#include <cstring>
#include <cwchar>
#include <omp.h>
#include <random>
#include <iostream>

#include "msort.h"


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
  const unsigned int t = atol(argv[2]);
  const unsigned int ts = atol(argv[3]);

  omp_set_num_threads(t);

  int *arr = new int[n];
  generate<int>(arr, n, -1000, 1000);

  double T0 = omp_get_wtime();
  msort(arr, n, ts);
  double T1 = omp_get_wtime();
  double ms = (T1 - T0) * 1000;

  printf("%d\n%d\n%f\n", arr[0], arr[n - 1], ms);

  delete[] arr;
  
  return 0;
}

