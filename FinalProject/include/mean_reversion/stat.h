#ifndef _MEAN_REVERSION_STAT_H_
#define _MEAN_REVERSION_STAT_H_

#include <cmath>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include <functional>
#include <type_traits>

#ifdef X_OPENMP
#include <omp.h>
#endif

namespace mean_reversion {
namespace stat {
namespace openmp {
    template<typename T>
    double reduce(const std::vector<T> &v, double init, std::function<double(double, T)> reducer);

    std::vector<double> mavg(const std::vector<double> &data, int window);

    std::vector<double> mstd(const std::vector<double> &data, int window);
} // namespace openmp

namespace cuda {
    
} // namespace cuda

inline namespace basic {
    /**
     * @brief Sums the vector of doubles.
     * 
     * @param v The vector of doubles.
     * @return The sum of the doubles.
     */
    double reduce(const std::vector<double> &v);

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
    V reduce(const std::vector<T> &v, V init, std::function<V(V, T)> reducer);

    /**
     * @brief Computes the mean value of the vector of doubles.
     * 
     * Note that 0 is returned if the input vector is empty.
     * 
     * @param v The vector of doubles.
     * @return The mean of the doubles.
     */
    double mean(const std::vector<double> &v);

    /**
     * @brief Computes the standard deviation of the vector of doubles.
     * 
     * Note that 0 is returned if the input vector is empty.
     * 
     * @param v The vector of doubles.
     * @return The standard deviation of the doubles. 
     */
    double stdev(const std::vector<double> &v);

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
    double covar(const std::vector<double> &v1, const std::vector<double> &v2);

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
    double corr(const std::vector<double> &v1, const std::vector<double> &v2);

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
    T* max(const std::vector<T> &v, int (*compar)(const T*, const T*));

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
                                    const std::vector<double> &v2);

    /**
     * @brief Computes the standard error of the sample.
     * 
     * @param v The sample vector.
     * @return The standard error of the sample. 
     */
    double stderror(const std::vector<double> &v);

    /**
     * @brief Computes the t-value given the estimated coefficient and its 
     * standard error.
     * 
     * @param coeff The estimated coefficient.
     * @param se_coeff The standard error of the estimated coefficient.
     * @return The t-value of the estimated coefficient.
     */
    double tvalue(double coeff, double se_coeff);

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
    double pvalue(double t);

    /**
     * @brief Computes the z-score for the given dataset.
     * 
     * @param data The dataset.
     * @return The z-score for each value in the dataset.
     */
    std::vector<double> zscore(const std::vector<double> &data);

    /**
     * @brief Computes the first difference of the given time series.
     * 
     * Note that an empty vector is returned if the given time series is empty 
     * or has only one observation.
     * 
     * @param v The time series to be differenced.
     * @return The first difference.
     */
    std::vector<double> diff(const std::vector<double> &v);

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
                                        double slope, double intercept);

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
            double *out_slope, double *out_intercept);

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
    std::pair<double, double> adfuller(const std::vector<double> &v);

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
    std::vector<double> mavg(const std::vector<double> &data, int window);

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
    std::vector<double> mstd(const std::vector<double> &data, int window);
} // namespace basic (sequential)
} // namespace stat
} // namespace mean_reversion

#endif