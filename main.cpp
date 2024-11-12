#include <cstddef>
#include <iostream>
#include <chrono>

#include "distribution_map.h"

int main() {
    distribution_map map{};

    for (size_t i = 0; i < 1; i++) {
        std::chrono::high_resolution_clock::time_point t_0 = std::chrono::high_resolution_clock::now();
        map.update_map();
        std::chrono::high_resolution_clock::time_point t_1 = std::chrono::high_resolution_clock::now();
        std::cout << (t_1 - t_0).count()/1e6 << "ms" << std::endl;
        
        map.draw_map();
    }

    return 0;
}