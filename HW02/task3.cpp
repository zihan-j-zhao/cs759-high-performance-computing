#include <chrono>
#include <random>
#include <cstring>
#include <iostream>

#include "matmul.h"

using namespace std::chrono;

#define T0() \
  start = high_resolution_clock::now()

#define T1() \
  end = high_resolution_clock::now(); \
  dur = duration_cast<duration<double, std::milli>>(end - start) \

high_resolution_clock::time_point start, end;
duration<double, std::milli> dur;

// Reference: FAQ/BestPractices/random_numbers.md
void generate(double *output, std::size_t n, double lower, double upper) {
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<double> dist(lower, upper);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      output[i * n + j] = dist(generator);
    }
  }
}
// ===== above =====

int main(int argc, char *argv[]) {
  unsigned int DIM = 1024;

  double *A = new double[DIM * DIM];
  double *B = new double[DIM * DIM];
  double *C = new double[DIM * DIM];
  generate(A, DIM, 0.0, 5.0);
  generate(B, DIM, 0.0, 5.0);

  std::cout << DIM << std::endl;

  std::memset((void *)C, 0, DIM * DIM * sizeof(double));
  T0();
  mmul1(A, B, C, DIM);
  T1();

  std::cout << dur.count() << std::endl;
  std::cout << C[DIM * DIM - 1] << std::endl;

  std::memset((void *)C, 0, DIM * DIM * sizeof(double));
  T0();
  mmul2(A, B, C, DIM);
  T1();

  std::cout << dur.count() << std::endl;
  std::cout << C[DIM * DIM - 1] << std::endl;

  std::memset((void *)C, 0, DIM * DIM * sizeof(double));
  T0();
  mmul3(A, B, C, DIM);
  T1();

  std::cout << dur.count() << std::endl;
  std::cout << C[DIM * DIM - 1] << std::endl;

  std::vector<double> _A, _B;
  _A.assign(A, A + DIM * DIM);
  _B.assign(B, B + DIM * DIM);
  std::memset((void *)C, 0, DIM * DIM * sizeof(double));
  T0();
  mmul4(_A, _B, C, DIM);
  T1();

  std::cout << dur.count() << std::endl;
  std::cout << C[DIM * DIM - 1] << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;
}
