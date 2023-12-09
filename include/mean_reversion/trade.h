#ifndef _MEAN_REVERSION_TRADE_H_
#define _MEAN_REVERSION_TRADE_H_

#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

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

        vector<double> prices() const {
            return this->m_prices;
        }

        vector<string> dates() const {
            return this->m_dates;
        }

        void append(std::string &date, double price) {
            this->m_dates.push_back(date);
            this->m_prices.push_back(price);
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