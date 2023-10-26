#include "count.cuh"

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts) {
  // Reference: https://thrust.github.io/doc/group__reductions_gad5623f203f9b3fdcab72481c3913f0e0.html
  thrust::device_vector<int> in_key(d_in.size());
  in_key = d_in;
  thrust::device_vector<int> in_val(d_in.size());
  thrust::fill(in_val.begin(), in_val.end(), 1);

  thrust::pair<int *, int *> new_end;
  new_end = thrust::reduce_by_key(in_key.begin(), in_key.end(), in_val.begin(), values.begin(), counts.begin());
  // ===== above ====

  values.resize(new_end.first - values.begin());
  counts.resize(new_end.second - counts.begin());
}
