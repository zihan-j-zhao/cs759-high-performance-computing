#ifndef _MEAN_REVERSION_TRADE_H_
#define _MEAN_REVERSION_TRADE_H_

#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>
#include <iostream>
#include <algorithm>

#ifdef X_OPENMP
#include <omp.h>
#endif

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

        vector<int> m_buys;
        vector<int> m_sells;

    public:
        StockPair(const Stock &a, const Stock &b, double corr)
            : m_a(a), m_b(b), m_spread(a - b), m_ratio(a / b), m_corr(corr), 
                m_metric(a), m_metric_option('a') {

        }

        string name() const {
            return m_a.name() + "_" + m_b.name();
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

        vector<double> buys(char stock) const {
            size_t count = this->m_a.dates().size();
            vector<double> buy_prices(count, 0);
            vector<double> prices = stock == 'a' ? m_a.prices() : m_b.prices();

            for (int idx : this->m_buys)
                buy_prices[idx] = prices[idx];

            return buy_prices;
        }

        void set_buys(const vector<int> &buys) {
            this->m_buys = buys;
        }

        vector<double> sells(char stock) const {
            size_t count = this->m_a.dates().size();
            vector<double> sells_prices(count, 0);
            vector<double> prices = stock == 'a' ? m_a.prices() : m_b.prices();

            for (int idx : this->m_sells)
                sells_prices[idx] = prices[idx];

            return sells_prices;
        }

        void set_sells(const vector<int> &sells) {
            this->m_sells = sells;
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

        file.close();
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

        #pragma omp parallel for
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = i + 1; j < count; ++j) {
                vector<double> a = stocks[i].prices();
                vector<double> b = stocks[j].prices();
                double corr = mean_reversion::stat::corr(a, b);
                if (corr <= bar) {
                    pairs.push_back(StockPair(stocks[i], stocks[j], corr));
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
     * @brief Creates a list of indices of dates to buy/sell stocks.
     * 
     * @param pair The pair of stocks whose metric has been computed.
     */
    void create_signals(StockPair &pair) {
        auto prices = pair.metric().prices();
        
        auto mavg_5 = mean_reversion::stat::mavg(prices, 5);
        auto mavg_20 = mean_reversion::stat::mavg(prices, 20);
        auto mstd_20 = mean_reversion::stat::mstd(prices, 20);

        // Compute zscores
        vector<double> zscores;
        size_t mavg_count = min(mavg_20.size(), mavg_5.size());
        for (size_t i = 0; i < mavg_count; ++i)
            zscores.push_back((mavg_5[i + 15] - mavg_20[i]) / mstd_20[i]);
        
        // Compute signals
        vector<int> sells, buys;
        size_t metric_count = pair.metric().size();
        for (size_t i = 0; i < metric_count; ++i) {
            if (i < 20) {
                sells.push_back(i);
                continue;
            }

            if (zscores[i - 20] >= 1)
                sells.push_back(i);
            if (zscores[i - 20] <= -1) 
                buys.push_back(i);
        }
        
        pair.set_buys(buys);
        pair.set_sells(sells);
    }

    /**
     * @brief Saves buy/sell times of both stocks to a csv file.
     * 
     * Assume that no stocks will have prices at 0. All 0's in the file stand
     * for no moves.
     * 
     * @param pair The pair of stocks whose signals have been computed.
     * @param filepath The filepath to which the data will be saved.
     */
    void save_to_csv(const StockPair &pair, string &filepath) {
        ofstream file(filepath, ofstream::out);
        if (!file.is_open()) {
            cout << "error opening csv file: " << filepath << endl;
            return;
        }

        string a_tick = pair.first().name();
        string b_tick = pair.second().name();
        string a_buy = a_tick + "_BUY";
        string a_sell = a_tick + "_SELL";
        string b_buy = b_tick + "_BUY";
        string b_sell = b_tick + "_SELL";
        file << "DATE," << a_buy << "," << a_sell << ",";
        file << b_buy << "," << b_sell << "\n";  // headers

        vector<string> dates = pair.first().dates();
        vector<double> a_bp = pair.buys('a');
        vector<double> a_sp = pair.sells('a');
        vector<double> b_bp = pair.buys('b');
        vector<double> b_sp = pair.sells('b');
        size_t count = dates.size();
        
        for (size_t i = 0; i < count; ++i) {
            file << dates[i] << "," << a_bp[i] << "," << a_sp[i] << ",";
            file << b_bp[i] << "," << b_sp[i] << "\n";
        }

        file.close();
    }
} // namespace basic (sequential)
} // namesapce trade
} // namespace mean_reversion


#endif