#include <cmath>
#include <ratio>
#include <chrono>
#include <random>
#include <cstring>
#include <iostream>

#include "scan.h"

// Reference: directly copied from my HW01/task6.cpp for numeric validation
bool is_numeric(const char *str) {
    size_t size = std::strlen(str);
    for (size_t i = 0; i < size; ++i) {
        if (!std::isdigit(str[i])) {
            return false;
        }
    }
    return true;
}
// ===== above =====

using namespace std::chrono;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: task1 n" << std::endl;
        return 1;
    }

    if (!is_numeric(argv[1])) {
        std::cout << "Usage: task1 n" << std::endl;
        return 1;
    }

    size_t n = std::atol(argv[1]);
    float *arr = new float[n];
    float *scanned = new float[n];

    // Reference: directly copied from FAQ/BestPractices/random_numbers.md for
    // randomization
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    for (size_t i = 0; i < n; ++i) {
        arr[i] = dist(generator);
    }
    // ===== above =====

    // Reference: directly copied from Assignments/general/timing.md for timing
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> duration_milli;

    start = high_resolution_clock::now();
    scan(arr, scanned, n);
    end = high_resolution_clock::now();
    duration_milli = duration_cast<duration<double, std::milli>>(end - start);
    // ===== above =====

    std::cout << duration_milli.count() << std::endl;
    std::cout << scanned[0] << std::endl;
    std::cout << scanned[n - 1] << std::endl;

    delete[] arr;
    delete[] scanned;
}
