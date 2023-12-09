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
    std::tie(idx_a, idx_b, corr) = trade::select_pair(stocks);
    std::cout << stocks[idx_a].name() << " : " << stocks[idx_b].name() << std::endl;
    std::cout << corr << std::endl;
    
    return 0;
}
