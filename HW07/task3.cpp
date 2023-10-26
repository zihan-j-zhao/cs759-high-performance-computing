#include <omp.h>

#include <iostream>

int fac(int n) {
  int t = 1;
  for (int i = 1; i <= n; ++i) {
    t *= i;
  }
  return t;
}

int main() {
  omp_set_num_threads(4);
  printf("Number of threads: 4\n");

  #pragma omp parallel
  {
    printf("I'm thread No. %d\n", omp_get_thread_num());
  }

  #pragma omp parallel for
  for (int i = 1; i <= 8; ++i) {
    printf("%d!=%d\n", i, fac(i));
  }
  
  return 0;
}
