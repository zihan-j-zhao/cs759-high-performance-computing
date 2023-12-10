#ifdef X_CUDA
#ifndef _MEAN_REVERSION_KERNEL_H_
#define _MEAN_REVERSION_KERNEL_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__
void reduce_diffsq_kernel(const thrust::device_vector<double> &v, double *result);

#endif // _MEAN_REVERSION_KERNEL_H_
#endif // X_CUDA