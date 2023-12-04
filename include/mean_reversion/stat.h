#ifndef _MEAN_REVERSION_STAT_H_
#define _MEAN_REVERSION_STAT_H_

#include <cmath>
#include <vector>
#include <functional>

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
    V reduce(const std::vector<T> &v, V init, std::function<V(V, T*)> reducer) {
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
     * @brief Computes the residuals of the given vectors.
     * 
     * Note that an empty vector is returned if two vectors have different 
     * sizes or are both empty.
     * 
     * @param v1 A vector of doubles.
     * @param v2 A vector of doubles.
     * @return The residuals of the two vectors.
     */
    std::vector<double> residual(const std::vector<double> &v1, 
                                    const std::vector<double> &v2) {
        if (v1.size() != v2.size() || v1.empty()) return {};

        std::vector<double> res;
        size_t count = v1.size();
        for (size_t i = 0; i < count; ++i) res.push_back(v2[i] - v1[i]);
        return res;
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
     * @brief Computes the t-value given the estimated coefficient and its 
     * standard error.
     * 
     * @param coeff The estimated coefficient.
     * @param se_coeff The standard error of the estimated coefficient.
     * @return The t-value of the estimated coefficient.
     */
    double tvalue(double coeff, double se_coeff) {
        return coeff / se_coeff;
    }

    // TODO: pvalue function

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

    /**
     * @brief Computes the standard errors of estimated slope and intercept.
     * 
     * Note that a pari of {0.0, 0.0} is returned if two vectors have different
     * sizes or are both empty.
     * 
     * @param x The independent values in a linear regression model.
     * @param y The dependent values in a linear regression model.
     * @param slope The estimated slope.
     * @param intercept The estimated intercept.
     * @return [0] stores the error for slope; [1] the error for intercept.
     */
    std::pair<double, double> stderrors(const std::vector<double> &x, 
                                        const std::vector<double> &y, 
                                        double slope, double intercept) {
        if (x.size() != y.size() || x.empty()) return {0.0, 0.0};

        // Compute the residual sum of squares (RSS)
        std::vector<double> y_pred;
        for (double xi : x) y_pred.push_back(slope * xi + intercept);
        std::vector<double> res = residual(y_pred, y);
        auto rss_reducer = [](double acc, const double *e) {
            return acc + std::pow(*e, 2);
        };
        double rss = reduce<double, double>(res, 0.0, rss_reducer);

        // Compute the sum of squares of differences
        size_t count = x.size();
        double mean_x = mean(x);
        auto sqdiff_reducer = [mean_x](double acc, const double *xi) {
            return acc + std::pow(*xi - mean_x, 2);
        };
        double sqdiff = reduce<double, double>(x, 0.0, sqdiff_reducer);

        // Compute the standard errors for slope and intercept
        double se_slope = std::sqrt(rss / (count - 2) / sqdiff);
        double se_intercept = se_slope;
        se_intercept *= std::sqrt(1 / count + std::pow(mean_x, 2) / sqdiff);
        return {se_slope, se_intercept};
    }

    /**
     * @brief Computes the t-value and p-value for the given dataset.
     * 
     * @param v The dataset.
     * @return [0] stores the t-value; [1] the p-value
     */
    std::pair<double, double> adfuller(const std::vector<double> &v) {
        return {0.0, 0.0};
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