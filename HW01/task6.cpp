#include <string>
#include <cstring>
#include <iostream>

bool is_numeric(const char *str) {
    size_t size = std::strlen(str);
    for (size_t i = 0; i < size; ++i) {
        if (!std::isdigit(str[i])) {
            std::cout << str[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: task6 N" << std::endl;
        return 1;
    }

    if (!is_numeric(argv[1])) {
        std::cout << "Usage: task6 N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    for (int i = 0; i <= N; ++i)  {
        if (i == N) {
            std::printf("%d\n", i);
        } else {
            std::printf("%d ", i);
        }
    }
    for (int i = N; i >= 0; --i) {
        if (i == 0) {
            std::cout << i << std::endl;
        } else {
            std::cout << i << " ";
        }
    }

    return 0;
}
