#include <algorithm>

#include "msort.h"

static void msort_helper(int *arr, const std::size_t n, const std::size_t threshold, int t) {
  if (n <= 1) {
    return;
  }

  if (n < threshold) {
    int *it = arr;
    int *end = arr + n;
    while (it != end) {
      std::rotate(std::upper_bound(arr, it, *it), it, it + 1);
      it++;
    }
    return;
  }

  unsigned int half_arr = n / 2;
  unsigned int half_threads = t / 2;
  if (t == 1) {
    msort_helper(arr, half_arr, threshold, 1);
    msort_helper(arr + half_arr, n - half_arr, threshold, 1);
  } else {
    #pragma omp task
    msort_helper(arr, half_arr, threshold, half_threads);
    #pragma omp task
    msort_helper(arr + half_arr, n - half_arr, threshold, t - half_threads);
    #pragma omp taskwait
  }

  std::inplace_merge(arr, arr + half_arr, arr + n);
}

void msort(int *arr, const std::size_t n, const std::size_t threshold) {
  #pragma omp parallel
  #pragma omp single
  msort_helper(arr, n, threshold, omp_get_num_threads());
}
