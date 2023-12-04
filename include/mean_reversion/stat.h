#ifndef _MEAN_REVERSION_STAT_H_
#define _MEAN_REVERSION_STAT_H_

#include <cmath>
#include <vector>

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
     * @brief Computes the mean value of the vector of doubles.
     * 
     * Note that if the input vector is empty, this returns 0.
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
     * Note that if the input vector is empty, this returns 0.
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
     * Note that if input vectors have different sizes or both are empty, this 
     * returns 0.
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
     * Note that if input vectors have different sizes or both are empty, this 
     * returns 0.
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
}

namespace openmp
{

}

namespace cuda
{

}
}
} // namespace stat

#endif