#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = 0; y < n; ++y) {
      for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
          std::size_t _x = x + i - (m - 1) / 2;
          std::size_t _y = y + j - (m - 1) / 2;
          float val = 0;
          if (_x >= 0 && _x < n && _y >= 0 && _y < n) {
            val = image[_x * n + _y];
          } else if ((_x < 0 || _x >= n) && (_y < 0 || _y >= n)) {
            val = 0;
          } else {
            val = 1;
          }
          output[x * n + y] += mask[i * m + j] * val;
        }
      }
    }
  }
}
