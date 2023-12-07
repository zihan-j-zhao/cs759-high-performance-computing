#include <iostream>

#include "mean_reversion/stat.h"

int main(int argc, char *argv[])
{
  std::cout << "Hello World\n";

  std::vector<double> d1 = {10, 2, 7, 1, 9, 6, 5, 8, 3, 4, 
                            10, 2, 7, 1, 9, 6, 5, 8, 3, 4, 
                            10, 2, 7, 1, 9, 6, 5, 8, 3, 4}; // stationary!
  std::vector<double> d2 = {1, 2, 3, 4, 5, 8};
  double t, p;
  std::tie(t, p) = mean_reversion::stat::adfuller(d1);
  std::cout << t << "," << p << std::endl;

  return 0;
}
