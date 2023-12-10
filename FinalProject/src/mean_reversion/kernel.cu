#include "mean_reversion/kernel.h"

#ifdef X_CUDA
__global__
void reduce_diffsq_kernel(const thrust::device_vector<double> &v, double mean, double *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    int count = v.size();
    double sum = 0.0;

    for (int i = 0; i < count; i += step)
        sum += (v[i] - mean) * (v[i] - mean);
    
    atomicAdd(result, sum);
}
#endif