#ifndef _MEAN_REVERSION_STAT_CUDA_H_
#define _MEAN_REVERSION_STAT_CUDA_H_

#include <cmath>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include <functional>
#include <type_traits>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "kernel.cuh"

#define BLOCK_SIZE 256

namespace mean_reversion {
namespace stat{
namespace cuda {
    double reduce(const std::vector<double> &v);
    
    double mean(const std::vector<double> &v);

    __host__ double stdev(const std::vector<double> &v);
} // namespace cuda
} // namespace stat
} // namespace mean_reversion
#endif
