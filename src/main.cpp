#include <iostream>

#include "mean_reversion/trade.h"

using namespace mean_reversion;

int main(int argc, char *argv[]) {
    // if (argc < 2) {
    //     std::cerr << "Usage: ./main [csv_file]" << std::endl;
    //     return 1;
    // }

    std::string filepath = "F:\\University of Wisconsin-Madison\\Computer Science 759\\FinalProject\\NASDAQ_Top_50_history.csv";
    std::vector<trade::Stock> stocks = trade::load_from_csv(filepath);

    int idx_a, idx_b;
    double corr;
    // trade::Stock metric(std::string("default"));
    auto pairs = trade::find_pairs(stocks, -.75);
    trade::sort_pairs(pairs);
    std::cout << pairs.size() << std::endl;
    // std::tie(idx_a, idx_b, corr) = pairs[0];
    // std::cout << stocks[idx_a].name() << " : " << stocks[idx_b].name() << std::endl;
    // std::cout << corr << std::endl;
    for (auto pair : pairs) {\
        if (trade::test_pair(pair)) {
            std::cout << "found: " << pair.metric().name() << std::endl;
        }
    }
    
    return 0;
}
