#include <cmath>
#include <cstring>
#include <cwchar>
#include <random>
#include <iostream>

#include "convolution.h"


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

  float *image, *mask, *output;
  image = new float[n * n];
  mask = new float[9];
  output = new float[n * n];

  generate<float>(image, n * n, -10.0f, 10.0f);
  generate<float>(mask, 9, -1.0f, 1.0f);
  memset(output, 0, n * n * sizeof(float));

  double T0 = omp_get_wtime();
  convolve(image, output, n, mask, 3);
  double T1 = omp_get_wtime();
  double ms = (T1 - T0) * 1000;

  printf("%f\n%f\n%f\n", output[0], output[n * n - 1], ms);

  delete[] image;
  delete[] mask;
  delete[] output;

  return 0;
}

