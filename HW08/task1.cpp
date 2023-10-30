#include <cmath>
#include <cstring>
#include <cwchar>
#include <random>
#include <iostream>

#include "matmul.h"


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
  const std::size_t n = atol(argv[1]);
  const std::size_t t = atol(argv[2]);

  omp_set_num_threads(t);

  float *A, *B, *C;
  A = new float[n * n];
  B = new float[n * n];
  C = new float[n * n];

  generate<float>(A, n * n, 0.0, 5.0);
  generate<float>(B, n * n, 0.0, 5.0);
  memset(C, 0, n * n * sizeof(float));

  double T0 = omp_get_wtime();
  mmul(A, B, C, n);
  double T1 = omp_get_wtime();
  double ms = (T1 - T0) * 1000;

  printf("%f\n%f\n%f\n", C[0], C[n * n - 1], ms);

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

