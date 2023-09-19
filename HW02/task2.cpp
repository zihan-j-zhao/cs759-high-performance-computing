#include <chrono>
#include <random>
#include <cstring>
#include <iostream>

#include "convolution.h" 

using namespace std::chrono;

// Reference: directly copied from my HW01/task6.cpp for numeric validation
bool is_numeric(const char *str) {
  size_t size = std::strlen(str);
  for (size_t i = 0; i < size; ++i) {
    if (!std::isdigit(str[i])) {
      return false;
    }
  }
  return true;
}
// ===== above =====

// Reference: FAQ/BestPractices/random_numbers.md
void generate(float *output, std::size_t n, float lower, float upper) {
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(lower, upper);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      output[i * n + j] = dist(generator);
    }
  }
}
// ===== above =====

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: task2 n m" << std::endl;
    return 1;
  }

  if (!is_numeric(argv[1]) || !is_numeric(argv[2])) {
    std::cout << "Usage: task2 n m" << std::endl;
    return 1;
  }

  std::size_t n = std::atol(argv[1]);
  std::size_t m = std::atol(argv[2]);
  float *image = new float[n * n];
  float *mask = new float[m * m];
  float *output = new float[n * n];
  generate(image, n, -10.0f, 10.0f);
  generate(mask, m, -1.0f, 1.0f);

  // Reference: Assignments/general/timing.md
  high_resolution_clock::time_point start, end;
  duration<double, std::milli> duration_milli;
  start = high_resolution_clock::now();
  convolve(image, output, n, mask, m);  // timed function
  end = high_resolution_clock::now();
  duration_milli = duration_cast<duration<double, std::milli>>(end - start);
  // ===== above =====
  
  std::cout << duration_milli.count() << std::endl;
  std::cout << output[0] << std::endl;
  std::cout << output[n * n - 1] << std::endl;

  delete[] image;
  delete[] mask;
  delete[] output;
}
