#include "mean_reversion/stat_cuda.h"

double cuda::reduce(const std::vector<double> &v) {
    size_t count = v.size();
    thrust::host_vector<double> hvec(count);
    for (size_t i = 0; i < count; ++i)
        hvec[i] = v[i];
    
    thrust::device_vector<double> dvec = hvec;
    return thrust::reduce(dvec.begin(), dvec.end(), 0.0, thrust::plus<double>());
}

__host__
double cuda::stdev(const std::vector<double> &v) {
    if (v.empty()) return 0.0;

    double diffsq = 0.0;
    size_t count = v.size();
    double mean = baisc::mean(v);
    thrust::host_vector<double> hvec(count);
    for (size_t i = 0; i < count; ++i)
        hvec[i] = v[i];
    
    thrust::device_vector<double> dvec = hvec;
    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    reduce_diffsq_kernel<<<num_blocks, BLOCK_SIZE>>>(devc, mean, &diffsq);
    return std::sqrt(diffsq / count);
}