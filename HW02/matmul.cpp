#include "matmul.h"

void mmul1(const double *A, const double *B, double *C, const unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int k = 0; k < n; ++k) {
        // [0] * [0] + [1] * [n] + [2] * [2n]
        // [n] * [1] + [n+1] * [n+1] + [n+2]*[2n+1]
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

void mmul2(const double *A, const double *B, double *C, const unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int k = 0; k < n; ++k) {
      for (unsigned int j = 0; j < n; ++j) {
        // [0] * [0] + [1] * [n] + [2] * [2n]
        // [n] * [1] + [n+1] * [n+1] + [n+2]*[2n+1]
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

void mmul3(const double *A, const double *B, double *C, const unsigned int n) {
  for (unsigned int j = 0; j < n; ++j) {
    for (unsigned int k = 0; k < n; ++k) {
      for (unsigned int i = 0; i < n; ++i) {
        // [0] * [0] + [1] * [n] + [2] * [2n]
        // [n] * [1] + [n+1] * [n+1] + [n+2]*[2n+1]
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

void mmul4(const std::vector<double> &A, const std::vector<double> &B, double *C, const unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int k = 0; k < n; ++k) {
        // [0] * [0] + [1] * [n] + [2] * [2n]
        // [n] * [1] + [n+1] * [n+1] + [n+2]*[2n+1]
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}
