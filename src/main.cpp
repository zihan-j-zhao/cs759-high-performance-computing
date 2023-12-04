#include <iostream>

#include "mean_reversion/stat.h"

int main(int argc, char *argv[])
{
  std::cout << "Hello World\n";

  std::vector<double> d1 = {1, 2, 3, 4, 5, 6};
  std::vector<double> d2 = {1, 2, 3, 4, 5, 6};
  double covar = mean_reversion::stat::covar(d1, d2);
  std::cout << covar << std::endl;
  double corr = mean_reversion::stat::corr(d1, d2);
  std::cout << corr << std::endl;

  return 0;
}
