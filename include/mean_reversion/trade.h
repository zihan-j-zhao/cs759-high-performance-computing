#ifndef _MEAN_REVERSION_TRADE_H_
#define _MEAN_REVERSION_TRADE_H_

#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "stat.h"

namespace mean_reversion {
namespace trade {
inline namespace basic {
    using namespace std;

    class Stock {
    private:
        string m_tick;  // stock symbol (e.g., AAPL for Apple Inc.)
        vector<double> m_prices;  // historical close prices
        vector<string> m_dates;   // corresponding dates
    public:
        Stock(string &name) : m_tick(name), m_prices({}), m_dates({}) {

        }

        Stock(string &name, vector<double> &prices, vector<string> &dates) 
            : m_tick(name), m_prices(prices), m_dates(dates) {
        }

        size_t size() const {
            return this->m_prices.size();
        }

        string name() const {
            return this->m_tick;
        }

        void set_name(const string &n) {
            this->m_tick = n;
        }

        vector<double> prices() const {
            return this->m_prices;
        }

        void set_prices(const vector<double> &p) {
            this->m_prices = p;
        }

        vector<string> dates() const {
            return this->m_dates;
        }

        void set_dates(const vector<string> &d) {
            this->m_dates = d;
        }

        void append(string &date, double price) {
            this->m_dates.push_back(date);
            this->m_prices.push_back(price);
        }

        Stock operator/(const Stock &b) const {
            size_t count = min(size(), b.size());

            string name = "Ratio (" + m_tick + " - " + b.m_tick + ")";
            vector<string> dates(m_dates.begin(), m_dates.begin() + count);
            vector<double> ratio;
            for (size_t i = 0; i < count; ++i)
                ratio.push_back(m_prices[i] / b.m_prices[i]);
            
            return Stock(name, ratio, dates);
        }

        Stock operator-(const Stock &b) const {
            size_t count = min(size(), b.size());

            std::string name = "Spread (" + m_tick + " - " + b.m_tick + ")";
            vector<string> dates(m_dates.begin(), m_dates.begin() + count);
            vector<double> spread;
            for (size_t i = 0; i < count; ++i)
                spread.push_back(m_prices[i] - b.m_prices[i]);
            
            return Stock(name, spread, dates);
        }
    };

    class StockPair {
    private:
        Stock m_a;
        Stock m_b;
        Stock m_spread;
        Stock m_ratio;
        double m_corr;

        Stock m_metric;
        char m_metric_option;

    public:
        StockPair(const Stock &a, const Stock &b, double corr)
            : m_a(a), m_b(b), m_spread(a - b), m_ratio(a / b), m_corr(corr), 
                m_metric(a), m_metric_option('a') {

        }

        Stock first() const {
            return this->m_a;
        }

        Stock second() const {
            return this->m_b;
        }

        Stock ratio() const {
            return this->m_ratio;
        }

        Stock spread() const {
            return this->m_spread;
        }

        double corr() const {
            return this->m_corr;
        }

        Stock metric() const {
            return this->m_metric;
        }

        char metric_option() const {
            return this->m_metric_option;
        }

        void set_metric(char option) {
            if (option == 'a') {
                this->m_metric = m_a;
                this->m_metric_option = 'a';
            } else if (option == 'b') {
                this->m_metric = m_b;
                this->m_metric_option = 'b';
            } else if (option == 'r') {
                this->m_metric = m_ratio;
                this->m_metric_option = 'r';
            } else {
                this->m_metric = m_spread;
                this->m_metric_option = 's';
            }
        }

        bool operator<(const StockPair &other) const {
            return m_corr < other.m_corr;
        }

        bool operator==(const StockPair &other) const {
            return (m_a.name() == other.m_a.name() && m_b.name() == other.m_b.name())
                || (m_a.name() == other.m_b.name() && m_b.name() == other.m_a.name());
        }
    };

    /**
     * @brief Loads historical stock data from a csv file.
     * 
     * This function assumes the csv file follows the format (without tabs):
     *      DATE,           TICKER1,    TICKER2,    TICKER3,    ...
     *      2022-12-01,     100,        200,        300,        ...
     *      ...
     * This means the first row is always the headers and the first column is 
     * always the dates in YYYY-MM-DD format. Moreover, it assumes the csv file
     * does not contain null values.
     * 
     * @param filepath The path to the csv file.
     * @return All the stock histories, as a vector of Stock objects.
     */
    vector<Stock> load_from_csv(const string &filepath) {
        ifstream file(filepath);
        if (!file.is_open()) {
            cout << "error opening csv file: " << filepath << endl;
            return {};
        }

        string line;
        vector<Stock> stocks;
        bool header = true;
        while (getline(file, line)) {
            int idx = 0;
            bool dates = true;
            std::string today, cell;
            stringstream stream(line);

            while (getline(stream, cell, ',')) {
                if (dates) {
                    dates = false;
                    today = cell;
                    continue;
                }

                if (header)
                    stocks.push_back(Stock(cell));
                else
                    stocks[idx++].append(today, stod(cell));
            }

            header = header && false;
        }

        return stocks;
    }

    /**
     * @brief Finds all pairs of stocks whose correlation is <= bar.
     * 
     * Note that an empty vector is returned on invalid inputs.
     * 
     * @param stocks The list of stocks.
     * @param bar The cap vaule for correlation coefficients.
     * @return The list of pairs of stocks that satisfy requirements.
     */
    vector<StockPair> find_pairs(const vector<Stock> &stocks, double bar) {
        if (stocks.empty() || bar < -1.0 || bar > 1.0) return {};

        vector<StockPair> pairs;
        size_t count = stocks.size();
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < count; ++j) {
                vector<double> a = stocks[i].prices();
                vector<double> b = stocks[j].prices();
                double corr = mean_reversion::stat::corr(a, b);
                if (corr <= bar) {
                    StockPair pair(stocks[i], stocks[j], corr);
                    auto lambda = [pair](const StockPair &other) {
                        return other == pair;
                    };
                    auto found = find_if(pairs.begin(), pairs.end(), lambda);
                    if (found == end(pairs)) {
                        pairs.push_back(pair);
                    }
                }
            }
        }

        return pairs;
    }

    /**
     * @brief Sorts tuples in place in ascending order of the third element.
     * 
     * A tuple includes two indices and a correlation coefficient. The sorting
     * is based on the correlation coefficient.
     * 
     * @param pairs The list of tuples to be sorted.
     */
    void sort_pairs(vector<StockPair> &pairs) {
        auto compar = [](StockPair a, StockPair b) {
            return a < b;
        };
        sort(pairs.begin(), pairs.end(), compar);
    }

    /**
     * @brief Tests whether the pair of stocks is suitable for trading.
     * 
     * @param pair The pair of stocks to be tested.
     * @param bar The cap value for its p-value.
     * @return true if the pair is suitable; false otherwise.
     */
    bool test_pair(StockPair &pair, double bar = 0.3) {
        auto a = pair.first();
        auto b = pair.second();
        auto r = pair.ratio();
        auto s = pair.spread();

        auto a_st = mean_reversion::stat::adfuller(a.prices());
        auto b_st = mean_reversion::stat::adfuller(b.prices());
        auto r_st = mean_reversion::stat::adfuller(r.prices());
        auto s_st = mean_reversion::stat::adfuller(s.prices());

        vector<double> stats = {a_st.second, b_st.second, r_st.second, s_st.second};
        auto min_pval = min_element(stats.begin(), stats.end());
        if (*min_pval <= bar) {
            if (*min_pval == stats[0]) {
                pair.set_metric('a');
            } else if (*min_pval == stats[1]) {
                pair.set_metric('b');
            } else if (*min_pval == stats[2]) {
                pair.set_metric('r');
            } else {
                pair.set_metric('s');
            }
            return true;
        }

        return false;
    }

    /**
     * @brief Selects a pair of stocks.
     * 
     * This function selects the pair whose correlation is the most negative,
     * i.e., whose PCC is closest to -1.0. The reason is that we want to long
     * stock A and short stock B. Note aslo that with this goal, it never
     * returns a pair of the same stock because (A, A) always has a PCC = 1.0.
     * (-1, -1) is returned if stocks is empty.
     * 
     * @param stocks The list of stocks to compare.
     * @return [0] and [1] store the indices of stocks; [2] the PCC.
     */
    tuple<int, int, double> select_pair(const vector<Stock> &stocks) {
        if (stocks.empty()) return {-1, -1, 1.0};

        double corr = 1.0; // max, corr in [-1.0, 1.0]
        int idx_a = -1, idx_b = -1;
        size_t count = stocks.size();

        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < count; ++j) {
                vector<double> a = stocks[i].prices();
                vector<double> b = stocks[j].prices();
                double i_j_corr = mean_reversion::stat::corr(a, b);
                if (i_j_corr <= corr) {
                    idx_a = i;
                    idx_b = j;
                    corr = i_j_corr;
                }
            }
        }

        return {idx_a, idx_b, corr};
    }

} // namespace basic (sequential)
} // namesapce trade
} // namespace mean_reversion


#endif