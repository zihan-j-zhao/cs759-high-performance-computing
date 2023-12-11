#include "mean_reversion/kernel.cuh"

__device__ 
static double __atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do { 
        assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__
void reduce_diffsq_kernel(const double *v, int n, double mean, double *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    double sum = 0.0;

    for (int i = idx; i < n; i += step)
        sum += (v[i] - mean) * (v[i] - mean);
    
    __atomicAdd(result, sum);
}
