#include <cuda.h>
#include <iostream>

#include "mean_reversion/trade.h"

using namespace mean_reversion;

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    std::string filepath = argv[1];
    std::vector<trade::Stock> stocks = trade::load_from_csv(filepath);

    auto pairs = trade::find_pairs(stocks, -.85);
    trade::sort_pairs(pairs);
    std::cout << pairs.size() << std::endl;
    // std::tie(idx_a, idx_b, corr) = pairs[0];
    // std::cout << stocks[idx_a].name() << " : " << stocks[idx_b].name() << std::endl;
    // std::cout << corr << std::endl;
    for (auto pair : pairs) {
        if (trade::test_pair(pair, 0.05)) {
            // std::cout << "found: " << pair.metric().name() << std::endl;
            // std::cout << "metric: " << pair.metric_option() << std::endl;
            // std::cout << "corr: " << pair.corr() << std::endl;

            std::string filepath = pair.name() + "_times.csv";
            trade::create_signals(pair);
            trade::save_to_csv(pair, filepath);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << ms << " ms" << std::endl;
    
    return 0;
}
