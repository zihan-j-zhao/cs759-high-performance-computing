#include <iostream>
#include <ratio>
#include <chrono>

#include "mean_reversion/trade.h"

using namespace mean_reversion;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./main [csv_file]" << std::endl;
        return 1;
    }

    #ifdef X_OPENMP
    std::cout << "Running OpenMP version" << std::endl;
    #endif

    #ifdef X_CUDA
    std::cout << "Running CUDA version" << std::endl;
    #endif

    high_resolution_clock::time_point start, end;
    duration<double, std::milli> duration_milli;

    start = high_resolution_clock::now();

    std::string filepath = argv[1];
    std::vector<trade::Stock> stocks = trade::load_from_csv(filepath);
    auto pairs = trade::find_pairs(stocks, -.85);
    trade::sort_pairs(pairs);
    std::cout << pairs.size() << std::endl;
    for (auto pair : pairs) {
        if (trade::test_pair(pair, 0.05)) {
            std::string filepath = pair.name() + "_times.csv";
            trade::create_signals(pair);
            trade::save_to_csv(pair, filepath);
        }
    }

    end = high_resolution_clock::now();
    duration_milli = duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_milli.count() << std::endl;
    
    return 0;
}
