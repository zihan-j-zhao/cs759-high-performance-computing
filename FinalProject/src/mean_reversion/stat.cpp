#include "mean_reversion/stat.h"

using namespace mean_reversion::stat;

#ifdef X_OPENMP
template<typename T>
double openmp::reduce(const std::vector<T> &v, double init, std::function<double(double, T)> reducer) {
    double acc = init;
    size_t count = v.size();

    #pragma omp parallel
    {
        double local_acc = init;

        #pragma omp for nowait
        for (size_t i = 0; i < count; ++i)
            local_acc = reducer(local_acc, v[i]);

        #pragma omp atomic
        acc += local_acc;
    }

    return acc;
}

std::vector<double> openmp::mavg(const std::vector<double> &data, int window) {
    if (data.empty() || window <= 0) return {};

    std::vector<double> average;
    int size = data.size();
    #pragma omp parallel 
    {
        std::vector<double> avg_private;

        #pragma omp for nowait schedule(static)
        for (int i = 0; i < size - window; ++i) {
            std::vector<double> part(data.begin() + i, data.begin() + i + window);
            avg_private.push_back(basic::mean(part));
        }

        #pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); ++i) {
            #pragma omp ordered
            average.insert(average.end(), avg_private.begin(), avg_private.end());
        }
    }

    return average;
}

std::vector<double> openmp::mstd(const std::vector<double> &data, int window) {
    if (data.empty() || window <= 0) return {};

    std::vector<double> average;
    int size = data.size();
    #pragma omp parallel 
    {
        std::vector<double> avg_private;

        #pragma omp for nowait schedule(static)
        for (int i = 0; i < size - window; ++i) {
            std::vector<double> part(data.begin() + i, data.begin() + i + window);
            avg_private.push_back(basic::stdev(part));
        }

        #pragma omp for schedule(static) ordered
        for (int i = 0; i < omp_get_num_threads(); ++i) {
            #pragma omp ordered
            average.insert(average.end(), avg_private.begin(), avg_private.end());
        }
    }

    return average;
}
#endif // X_OPENMP

double basic::reduce(const std::vector<double> &v) {
    #if X_CUDA
    return cuda::reduce(v);
    #endif

    double acc = 0;
    size_t count = v.size();

    #pragma omp parallel for reduction(+:acc)
    for (size_t i = 0; i < count; ++i)
        acc += v[i];

    return acc;
}

template<typename T, typename V>
V basic::reduce(const std::vector<T> &v, V init, std::function<V(V, T)> reducer) {
    #ifdef X_OPENMP
    if constexpr (std::is_same<V, double>::value)
        return openmp::reduce(v, init, reducer);
    #endif

    V acc = init;
    for (T e : v) acc = reducer(acc, e);
    return acc;
}

double basic::mean(const std::vector<double> &v) {
    if (v.empty()) return 0.0;

    size_t count = v.size();
    double acc = basic::reduce(v);
    return acc / count;
}

double basic::stdev(const std::vector<double> &v) {
    #ifdef X_CUDA
    return cuda::stdev(v);
    #endif
    
    if (v.empty()) return 0.0;

    double m = basic::mean(v);
    size_t count = v.size();
    double sq_diff = 0.0;
    auto reducer = [m](double acc, double e) {
        return acc + std::pow(e - m, 2);
    };
    sq_diff = basic::reduce<double, double>(v, sq_diff, reducer);
    return std::sqrt(sq_diff / count);
}

double basic::covar(const std::vector<double> &v1, const std::vector<double> &v2) {
    if (v1.size() != v2.size() || v1.empty()) 
        return 0.0;

    double mean1 = basic::mean(v1);
    double mean2 = basic::mean(v2);

    double cov = 0.0;
    size_t count = v1.size();

    #pragma omp parallel for reduction(+:cov)
    for (size_t i = 0; i < count; ++i)
        cov += (v1[i] - mean1) * (v2[i] - mean2);

    return cov / count;
}

double basic::corr(const std::vector<double> &v1, const std::vector<double> &v2) {
    if (v1.size() != v2.size() || v1.empty())
        return 1.0;

    return basic::covar(v1, v2) / (basic::stdev(v1) * basic::stdev(v2));
}

template<typename T>
T* basic::max(const std::vector<T> &v, int (*compar)(const T*, const T*)) {
    if (v.empty()) return nullptr;

    T* max_ele = &v[0];
    size_t count = v.size();
    
    #pragma omp parallel for reduction(max:max_ele)
    for (size_t i = 1; i < count; ++i)
        max_ele = compar(&v[i], max_ele) >= 1 ? &v[i] : max_ele;

    return max_ele;
}

std::vector<double> basic::residual(const std::vector<double> &v1, 
                                    const std::vector<double> &v2) {
    if (v1.size() != v2.size() || v1.empty()) return {};

    std::vector<double> res;
    size_t count = v1.size();
    for (size_t i = 0; i < count; ++i) 
        res.push_back(v2[i] - v1[i]);

    return res;
}

double basic::stderror(const std::vector<double> &v) {
    if (v.empty()) return 0.0;

    return basic::stdev(v) / v.size();
}

double basic::tvalue(double coeff, double se_coeff) {
    return coeff / se_coeff;
}

double basic::pvalue(double t) {
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
    #pragma omp parallel for reduction(+:temp)
    for (int i = tau_coeff.size() - 1; i >= 0; --i)
        temp += tau_coeff[i] * std::pow(t, i);

    // Use normal CDF
    return std::erfc(-temp / std::sqrt(2)) / 2.0;
}

std::vector<double> basic::zscore(const std::vector<double> &data) {
    if (data.empty()) return {};

    std::vector<double> scores;
    double _mean = basic::mean(data);
    double _stdev = basic::stdev(data);
    for (double val : data) 
        scores.push_back((val - _mean) / _stdev);

    return scores;
}

std::vector<double> basic::diff(const std::vector<double> &v) {
    if (v.empty() || v.size() < 2) return {};

    std::vector<double> d;
    size_t count = v.size();
    for (size_t i = 1; i < count; ++i)
        d.push_back(v[i] - v[i - 1]);
    
    return d;
}

std::pair<double, double> basic::stderrors(const std::vector<double> &x, 
                                        const std::vector<double> &y, 
                                        double slope, double intercept) {
    if (x.size() != y.size() || x.empty()) return {0.0, 0.0};

    // Compute the residual sum of squares (RSS)
    std::vector<double> y_pred;
    for (double xi : x) y_pred.push_back(slope * xi + intercept);
    std::vector<double> res = basic::residual(y_pred, y);
    auto rss_reducer = [](double acc, const double e) {
        return acc + std::pow(e, 2);
    };
    double rss = basic::reduce<double, double>(res, 0.0, rss_reducer);

    // Compute the sum of squares of differences
    size_t count = x.size();
    double mean_x = basic::mean(x);
    auto sqdiff_reducer = [mean_x](double acc, const double xi) {
        return acc + std::pow(xi - mean_x, 2);
    };
    double sqdiff = basic::reduce<double, double>(x, 0.0, sqdiff_reducer);

    // Compute the standard errors for slope and intercept
    double se_slope = std::sqrt(rss / (count - 2) / sqdiff);
    double se_intercept = se_slope;
    se_intercept *= std::sqrt(1 / count + std::pow(mean_x, 2) / sqdiff);
    return {se_slope, se_intercept};
}

int basic::ols(const std::vector<double> &x, const std::vector<double> &y, 
            double *out_slope, double *out_intercept) {
    if (x.size() != y.size() || x.empty() || x.size() < 2)  return -1; // InvalidInput

    size_t size = x.size();
    double sum_x = basic::reduce(x);
    double sum_y = basic::reduce(y);
    auto ssq_reducer = [](double acc, const double v) {
        return acc + std::pow(v, 2);
    };
    double sum_x2 = basic::reduce<double, double>(x, 0.0, ssq_reducer);
    double sum_y2 = basic::reduce<double, double>(y, 0.0, ssq_reducer);
    double sum_xy = 0.0;
    for (size_t i = 0; i < size; ++i) sum_xy += x[i] * y[i];

    double denom = (double) size * sum_x2 - std::pow(sum_x, 2);
    if (denom == 0) {
        *out_slope = 0;
        *out_intercept = 0;
        return 1; // Singular matrix
    }

    *out_slope = (size * sum_xy - sum_x * sum_y) / denom;
    *out_intercept = (sum_y - *out_slope * sum_x) / denom;

    return 0; // Success
}

std::pair<double, double> basic::adfuller(const std::vector<double> &v) {
    if (v.empty()) return {0.0, 0.0};

    // Compute the estimated slope and intercept coefficients
    double slope, intercept;
    std::vector<double> diff_v = basic::diff(v);
    std::vector<double> x(diff_v.begin(), diff_v.end() - 1);
    std::vector<double> y(diff_v.begin() + 1, diff_v.end());
    int error = basic::ols(x, y, &slope, &intercept);
    if (error != 0) {
        std::cout << "ols error occurred: " << error << std::endl;
        return {0.0, 0.0}; // Error!
    }

    // Compute t-value for the estimated slope and p-value based on t-value
    double se_slope, se_intercept;
    std::tie(se_slope, se_intercept) = basic::stderrors(x, y, slope, intercept);
    double t_value = basic::tvalue(slope, se_slope);
    double p_value = basic::pvalue(t_value);

    return {t_value, p_value};
}

std::vector<double> basic::mavg(const std::vector<double> &data, int window) {
    #ifdef X_OPENMP
    return openmp::mavg(data, window);
    #endif

    if (data.empty() || window <= 0) return {};

    std::vector<double> average;
    int size = data.size();

    for (int i = 0; i < size - window; ++i) {
        std::vector<double> part(data.begin() + i, data.begin() + i + window);
        average.push_back(basic::mean(part));
    }

    return average;
}

std::vector<double> basic::mstd(const std::vector<double> &data, int window) {
    #ifdef X_OPENMP
    return openmp::mstd(data, window);
    #endif

    if (data.empty() || window <= 0) return {};

    std::vector<double> stds;
    int size = data.size();
    for (int i = 0; i < size - window; ++i) {
        std::vector<double> part(data.begin() + i, data.begin() + i + window);
        stds.push_back(basic::stdev(part));
    }

    return stds;
}
