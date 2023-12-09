#ifndef _MEAN_REVERSION_STAT_H_
#define _MEAN_REVERSION_STAT_H_

#include <cmath>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
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
        for (T e : v) acc = reducer(acc, &e);
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

    /**
     * @brief Compute the p-value of the given test statistic.
     * 
     * Due to the assumption that the test statistic (or t-value) is computed
     * from the ADF test, calculations below are almost hardcoded for clarity.
     * 
     * References:
     * 1. Python statsmodels implementations of p-value computation
     *    (https://github.com/statsmodels/statsmodels/blob/main/statsmodels/
     *     tsa/adfvalues.py)
     * 2. StackOverflow post about approximating the normal distribution CDF
     *    (https://stackoverflow.com/questions/2328258/cumulative-normal-
     *     distribution-function-in-c-c)
     * 
     * @param t The test statistic.
     * @return The p-value.
     */
    double pvalue(double t) {
        static const std::vector<double> tau_max = {
            2.74, 0.92, 0.55, 0.61, 0.79, 1
        };
        static const std::vector<double> tau_min = {
            -18.83, -18.86, -23.48, -28.07, -25.96, -23.27
        };
        static const std::vector<double> tau_star = {
            -1.61, -2.62, -3.13, -3.47, -3.78, -3.93
        };
        static const std::vector<std::vector<double>> tau_smallp = {
            {2.1659, 1.4412, 0.038269},
            {2.92, 1.5012, 0.039796},
            {3.4699, 1.4856, 0.03164},
            {3.9673, 1.4777, 0.026315},
            {4.5509, 1.5338, 0.029545},
            {5.1399, 1.6036, 0.034445},
        };
        static const std::vector<std::vector<double>> tau_largep = {
            {0.4797, 0.93557, -0.06999, 0.033066},
            {1.5578, 0.8558, -0.2083, -0.033549},
            {2.2268, 0.68093, -0.32362, -0.054448},
            {2.7654, 0.64502, -0.30811, -0.044946},
            {3.2684, 0.68051, -0.26778, -0.034972},
            {3.7268, 0.7167, -0.23648, -0.028288},
        };

        if (t > tau_max[0]) return 1;
        else if (t < tau_min[0]) return 0.0;
        
        std::vector<double> tau_coeff;
        if (t <= tau_star[0]) tau_coeff = tau_smallp[0];
        else tau_coeff = tau_largep[0];

        double temp = 0;
        for (int i = tau_coeff.size() - 1; i >= 0; --i) {
            temp += tau_coeff[i] * std::pow(t, i);
        }

        // Use normal CDF
        return std::erfc(-temp / std::sqrt(2)) / 2.0;
    }

    /**
     * @brief Computes the z-score for the given dataset.
     * 
     * @param data The dataset.
     * @return The z-score for each value in the dataset.
     */
    std::vector<double> zscore(const std::vector<double> &data) {
        if (data.empty()) return {};

        std::vector<double> scores;
        double _mean = mean(data);
        double _stdev = stdev(data);
        for (double val : data) 
            scores.push_back((val - _mean) / _stdev);

        return scores;
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

    /**
     * @brief Computes the standard errors of estimated slope and intercept.
     * 
     * Note that a pari of {0.0, 0.0} is returned if two vectors have different
     * sizes or are both empty.
     * 
     * References:
     * 1. https://www.statology.org/standard-error-of-regression-slope/
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
     * @brief Computes the estimated slope and intercept of the given datasets
     * using the simple ordinary least squares (OLS) model.
     * 
     * The below algorithm uses a set of formulas documented in the "Simple 
     * Linear Regression Model" section of the OLS Wikipedia page
     * (https://en.wikipedia.org/wiki/Ordinary_least_squares). The following
     * implementation ignores the calculation of r2-value for simplicity, but
     * it is highly recommended to use the r2-value in otherwise situations.
     * 
     * @param x The independent dataset.
     * @param y The dependent dataset.
     * @param out_slope The variable storing the estimated slope.
     * @param out_intercept The variable storing the estimated intercept.
     * @return 0 for no errors; -1 for invalid inputs; 1 for singular matrix.
     */
    int ols(const std::vector<double> &x, const std::vector<double> &y, 
            double *out_slope, double *out_intercept) {
        if (x.size() != y.size() || x.empty() || x.size() < 2)  return -1; // InvalidInput

        size_t size = x.size();
        double sum_x = reduce(x);
        double sum_y = reduce(y);
        auto ssq_reducer = [](double acc, const double *v) {
            return acc + std::pow(*v, 2);
        };
        double sum_x2 = reduce<double, double>(x, 0.0, ssq_reducer);
        double sum_y2 = reduce<double, double>(y, 0.0, ssq_reducer);
        double sum_xy = 0.0;
        for (size_t i = 0; i < size; ++i) sum_xy += x[i] * y[i];

        double denom = (double) size * sum_x2 - std::pow(sum_x, 2);
        if (denom == 0) {
            *out_slope = 0;
            *out_intercept = 0;
            return 1; // Singular matrix
        }

        *out_slope = (size * sum_xy - sum_x * sum_y) / denom;
        *out_intercept = (sum_y * sum_x2 - sum_xy * sum_x) / denom;

        return 0; // Success
    }

    /**
     * @brief Computes the t-value and p-value for the given dataset.
     * 
     * This function implements a sipmlified version of the ADF test 
     * demonstrated on the Wikipedia page (https://en.wikipedia.org/wiki/
     * Augmented_Dickey%E2%80%93Fuller_test).
     * 
     * @param v The dataset.
     * @return [0] stores the t-value; [1] the p-value
     */
    std::pair<double, double> adfuller(const std::vector<double> &v) {
        if (v.empty()) return {0.0, 0.0};

        // Compute the estimated slope and intercept coefficients
        double slope, intercept;
        std::vector<double> diff_v = diff(v);
        std::vector<double> x(diff_v.begin(), diff_v.end() - 1);
        std::vector<double> y(diff_v.begin() + 1, diff_v.end());
        int error = ols(x, y, &slope, &intercept);
        if (error != 0) {
            std::cout << "ols error occurred: " << error << std::endl;
            return {0.0, 0.0}; // Error!
        }

        // Compute t-value for the estimated slope and p-value based on t-value
        double se_slope, se_intercept;
        std::tie(se_slope, se_intercept) = stderrors(x, y, slope, intercept);
        double t_value = tvalue(slope, se_slope);
        double p_value = pvalue(t_value);

        return {t_value, p_value};
    }

    /**
     * @brief Computes the moving average of the given dataset.
     * 
     * Note that an empty vector is returned if data is empty or window size
     * is invalid.
     * 
     * @param data The dataset.
     * @param window The window size (positive integer).
     * @return The moving average given the dataset and window size.
     */
    std::vector<double> mavg(const std::vector<double> &data, int window) {
        if (data.empty() || window <= 0) return {};

        std::vector<double> average;
        int size = data.size();
        for (int i = 0; i < size - window; ++i) {
            std::vector<double> part(data.begin() + i, data.begin() + i + window);
            average.push_back(mean(part));
        }

        return average;
    }

    /**
     * @brief Computes the moving standard deviation of the given dataset.
     * 
     * Note that an empty vector is returned if data is empty or window size
     * is invalid.
     * 
     * @param data The dataset.
     * @param window The window size (positive integer).
     * @return The moving standard deviation given the dataset and window size.
     */
    std::vector<double> mstd(const std::vector<double> &data, int window) {
        if (data.empty() || window <= 0) return {};

        std::vector<double> stds;
        int size = data.size();
        for (int i = 0; i < size - window; ++i) {
            std::vector<double> part(data.begin() + i, data.begin() + i + window);
            stds.push_back(stdev(part));
        }

        return stds;
    }
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