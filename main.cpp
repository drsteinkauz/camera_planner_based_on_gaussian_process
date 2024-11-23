#include <array>
#include <cstddef>
#include <iostream>
#include <chrono>

#include "distribution_map.h"

int main() {
    distribution_map map{};

    const int step_num = 200;
    const int traj_num = 30;
    std::array<std::array<double, 3>,step_num + traj_num> whole_traj{};
    double speed = 1.0;
    for (size_t i = 0; i < step_num + traj_num; i++) {
        whole_traj[i] = {static_cast<double>(i) * speed*map.step_time, static_cast<double>(i) * speed*map.step_time, 0.0};
    }
    double actual_camera_orientation{};

    for (size_t i = 0; i < step_num; i++) {
        map.robot_crt_position = whole_traj[i];
        map.camera_crt_orientation = actual_camera_orientation;
        for (size_t j = 0; j < traj_num; j++) {
            map.traj_waypt[j] = whole_traj[i+j];
        }
        
        // std::chrono::high_resolution_clock::time_point t_0 = std::chrono::high_resolution_clock::now();
        map.update_map();
        // std::chrono::high_resolution_clock::time_point t_1 = std::chrono::high_resolution_clock::now();
        // std::cout << (t_1 - t_0).count()/1e6 << "ms" << std::endl;

        actual_camera_orientation = map.choose_new_camera_orientation();
        
        map.draw_map();
    }

    return 0;
}