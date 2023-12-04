#ifndef _MEAN_REVERSION_STAT_H_
#define _MEAN_REVERSION_STAT_H_

#include <cmath>
#include <vector>

#ifdef X_OPENMP
#include <omp.h>
#endif

namespace mean_reversion {
namespace stat {
inline namespace basic {
    /**
     * @brief Sums the vector of doubles.
     * 
     * @param v The vector of doubles.
     * @return The sum of the doubles.
     */
    double reduce(const std::vector<double> &v) {
        double acc = 0;
        for (double e : v) acc += e;
        return acc;
    }

    /**
     * @brief Reduces the vector given the reducer.
     * 
     * @tparam T The type of elements in the vector.
     * @tparam V The type of the accumulator and return value.
     * @param v The vector to be reduced.
     * @param init The initial value of the accumulator.
     * @param reducer A user-supplied reducer function. 
     * @return V The final accumulator.
     */
    template<typename T, typename V>
    V reduce(const std::vector<T> &v, V init, V (*reducer)(V, const T*)) {
        V acc = init;
        for (const T &e : v) acc = reducer(acc, &e);
        return acc;
    }

    /**
     * @brief Computes the mean value of the vector of doubles.
     * 
     * Note that 0 is returned if the input vector is empty.
     * 
     * @param v The vector of doubles.
     * @return The mean of the doubles.
     */
    double mean(const std::vector<double> &v) {
        if (v.empty()) return 0.0;

        size_t count = v.size();
        double acc = reduce(v);
        return acc / count;
    }

    /**
     * @brief Computes the standard deviation of the vector of doubles.
     * 
     * Note that 0 is returned if the input vector is empty.
     * 
     * @param v The vector of doubles.
     * @return The standard deviation of the doubles. 
     */
    double stdev(const std::vector<double> &v) {
        if (v.empty()) return 0.0;

        double m = mean(v);
        size_t count = v.size();
        double sq_diff = 0.0;
        for (size_t i = 0; i < count; ++i)
            sq_diff += (v[i] - m) * (v[i] - m);
        
        return std::sqrt(sq_diff / count);
    }

    /**
     * @brief Computes the covariance of two vectors of doubles.
     * 
     * Note that 0 is returned if input vectors have different sizes or both 
     * are empty.
     * 
     * @param v1 A vector of doubles.
     * @param v2 A vector of doubles.
     * @return The covariance of two vectors. 
     */
    double covar(const std::vector<double> &v1, const std::vector<double> &v2) {
        if (v1.size() != v2.size() || v1.empty()) 
            return 0.0;

        double mean1 = mean(v1);
        double mean2 = mean(v2);

        double cov = 0.0;
        size_t count = v1.size();
        for (size_t i = 0; i < count; ++i)
            cov += (v1[i] - mean1) * (v2[i] - mean2);

        return cov / count;
    }

    /**
     * @brief Computes the Pearson correlation of two vectors of doubles.
     * 
     * The used formula for computing Pearson correlation, or Pearson 
     * Correlation Coefficient (PCC), is from its Wikipedia page
     * (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
     * 
     * Note that 0 is returned if input vectors have different sizes or both 
     * are empty.
     * 
     * @param v1 A vector of doubles.
     * @param v2 A vector of doubles.
     * @return The correlation of two vectors. 
     */
    double corr(const std::vector<double> &v1, const std::vector<double> &v2) {
        if (v1.size() != v2.size() || v1.empty())
            return 1.0;

        return covar(v1, v2) / (stdev(v1) * stdev(v2));
    }

    /**
     * @brief Gets the maximum element from the vector given the comparator.
     * 
     * Note that nullptr is returned if the input vector is empty.
     * 
     * @tparam T The type of elements in the vector.
     * @param v The vector from which the maximum element is extracted.
     * @param compar A user-defined comparator for finding the maximum element.
     * @return T The maximum element of the vector.
     */
    template<typename T>
    T* max(const std::vector<T> &v, int (*compar)(const T*, const T*)) {
        if (v.empty()) return nullptr;

        T* max_ele = &v[0];
        size_t count = v.size();
        for (size_t i = 1; i < count; ++i) {
            if (compar(&v[i], max_ele) >= 1) {
                max_ele = &v[i];
            }
        }

        return max_ele;
    }

    /**
     * @brief Computes the standard error of the sample.
     * 
     * @param v The sample vector.
     * @return The standard error of the sample. 
     */
    double stderror(const std::vector<double> &v) {
        if (v.empty()) return 0.0;

        return stdev(v) / v.size();
    }

    /**
     * @brief Computes the first difference of the given time series.
     * 
     * Note that an empty vector is returned if the given time series is empty 
     * or has only one observation.
     * 
     * @param v The time series to be differenced.
     * @return The first difference.
     */
    std::vector<double> diff(const std::vector<double> &v) {
        if (v.empty() || v.size() < 2) return {};

        std::vector<double> d;
        size_t count = v.size();
        for (size_t i = 1; i < count; ++i)
            d.push_back(v[i] - v[i - 1]);
        
        return d;
    }

    // TODO: ADF linear regression model (OLS, without time trend)
    // TODO: ADF critical value table
    // TODO: test statistic: gamma / stderror
} // namespace basic (sequential)

namespace openmp {
    // TODO: speed up 'for' loops
} // namespace openmp

namespace cuda {
    // TODO: speed up computation using CUDA
} // namespace cuda
} // namespace stat
} // namespace mean_reversion

#endif