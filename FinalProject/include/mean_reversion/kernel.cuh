#ifndef _MEAN_REVERSION_KERNEL_H_
#define _MEAN_REVERSION_KERNEL_H_

#include <cuda.h>

__global__ void reduce_diffsq_kernel(const double *v, double n, double mean, double *result);

#endif // _MEAN_REVERSION_KERNEL_H_
