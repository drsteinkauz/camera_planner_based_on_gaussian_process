#ifndef _DISTRIBUTION_MAP_H_
#define _DISTRIBUTION_MAP_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/hal/interface.h>
#include <vector>
#include <iostream>
#include <chrono>

struct distribution_map
{
    /*/
        Coordination: World Frame
        Units: meter, radian, second
    //*/
    
    static constexpr double step_time = 0.1;

    static constexpr double prior_prob = 0.2;
    static constexpr double stdev_vel = 2.0;
    static constexpr double stdev_acc = 10.0;
    
    static constexpr double grid_size = 0.1;
    static constexpr double map_size_hf = 10.0;
    static constexpr double kern_size_hf = 3.0 * stdev_vel * step_time;
    static constexpr int map_size_hf_dscrt = static_cast<int>(map_size_hf / grid_size);
    static constexpr int map_size_dscrt = 2 * map_size_hf_dscrt + 1;
    static constexpr int kern_size_hf_dscrt = static_cast<int>(kern_size_hf / grid_size);
    static constexpr int kern_size_dscrt = 2 * kern_size_hf_dscrt + 1;

    static constexpr double fov_angle = M_PI / 180.0 * 72.0;
    static constexpr double fov_depth = 5.0;
    static constexpr double fov_depth_min = 0.5;

    static constexpr int waypt_num = 30;
    static constexpr int waypt_start_idx = 5;
    static constexpr int waypt_interval = 5;

    static constexpr double dyn_obj_vanish_dist = 15.0;

    static constexpr double hybrid_urgency_grad = 1.0 / (M_PI - fov_angle/2.0);



    cv::Mat map_potential = cv::Mat::ones(map_size_dscrt, map_size_dscrt, CV_64F) * prior_prob;
    struct dyn_obj{
        double posi[2]{};
        double vel[2]{};
        double radius{};
        int frame_idx{};
    };
    std::vector<dyn_obj> known_obj{};



    cv::Mat urgency_map_potential = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_64F);
    std::vector<double> urgency_known_obj{};

    
    
    cv::Mat cvlt_kern = cv::Mat::zeros(kern_size_dscrt, kern_size_dscrt, CV_64F);
    std::vector<std::vector<double>> obs_obj_cost{}; // for hungarian algorithm
    std::vector<cv::Mat> pre_calculated_urgency_trajpt{};



    std::array<double, 3> robot_last_position{};
    double camera_last_orientation{};
    std::vector<dyn_obj> last_obs_dyn_obj{};
    cv::Mat last_obs_stat_obj = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_8UC1);

    std::array<double, 3> robot_crt_position{};
    double camera_crt_orientation{};
    std::vector<dyn_obj> crt_obs_dyn_obj{};
    cv::Mat crt_obs_stat_obj = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_8UC1);
    
    std::array<std::array<double, 3>, waypt_num> traj_waypt{}; // traj_waypt[0] is the next way point (expected robot new position)



    distribution_map()
    {
        // initialize convolute kernel
        double stdev_pos = stdev_vel * step_time;
        double sum_kern = 0.0;
        for (size_t i = 0; i < kern_size_dscrt; i++){
            for (size_t j = 0; j < kern_size_dscrt; j++){
                /*/
                double dist_sqr = std::pow((i - kern_size_hf_dscrt)*grid_size, 2) + std::pow((j - kern_size_hf_dscrt)*grid_size, 2);
                cvlt_kern.at<double>(i, j) = 1.0 / (2.0 * M_PI * std::pow(stdev_pos, 2)) * std::exp(-dist_sqr / (2.0 * std::pow(stdev_pos, 2))); // probalistic density
                //*/
                double dist_sqr = (i - kern_size_hf_dscrt)*grid_size * (i - kern_size_hf_dscrt)*grid_size + (j - kern_size_hf_dscrt)*grid_size * (j - kern_size_hf_dscrt)*grid_size;
                cvlt_kern.at<double>(i, j) = std::exp(-dist_sqr / (2.0 * stdev_pos * stdev_pos));
                sum_kern += cvlt_kern.at<double>(i, j);
            }
        }
        cvlt_kern /= sum_kern; // normalization

        // calculate urgency map for single trajectory points
        for (size_t i = static_cast<size_t>(waypt_start_idx); i + waypt_interval < waypt_num; i += static_cast<size_t>(waypt_interval)) {
            double urgency_radius = 3 * stdev_vel * i * step_time;
            int urgency_radius_dscrt = static_cast<int>(urgency_radius / grid_size);

            cv::Mat urgency_trajpt = cv::Mat::zeros(2*urgency_radius_dscrt+1, 2*urgency_radius_dscrt+1, CV_64F);
            generate_urgency_trajpt_matrix(&urgency_trajpt, urgency_radius_dscrt, i);

            pre_calculated_urgency_trajpt.push_back(urgency_trajpt);
        }
    }



    double theta_property(double theta)
    {
        if (theta > M_PI)
            return theta_property(theta - 2 * M_PI);
        else if (theta <= -M_PI)
            return theta_property(theta + 2 * M_PI);
        else
            return theta;
    }

    bool is_blocked_grid(cv::Mat const& ocp_grid_map, std::array<int, 2> const& obs_pt_idx, std::array<int, 2> const& cmr_pt_idx)
    {
        std::array<int, 2> srch_drct{cmr_pt_idx[0] - obs_pt_idx[0], cmr_pt_idx[1] - obs_pt_idx[1]};

        int srch_step_x = 0;
        int srch_step_y = 0;
        if (srch_drct[0] != 0)
            srch_step_x = srch_drct[0] / std::abs(srch_drct[0]);
        if (srch_drct[1] != 0)
            srch_step_y = srch_drct[1] / std::abs(srch_drct[1]);

        int x_crt_idx = obs_pt_idx[0];
        int y_crt_idx = obs_pt_idx[1];

        while (x_crt_idx != cmr_pt_idx[0] || y_crt_idx != cmr_pt_idx[1]) {
            if (ocp_grid_map.at<uchar>(x_crt_idx, y_crt_idx) == 1)
                return true;

            std::array<int, 2> heading_drct{cmr_pt_idx[0] - x_crt_idx, cmr_pt_idx[1] - y_crt_idx};
            if (srch_drct[0] * heading_drct[1] - srch_drct[1] * heading_drct[0] > 0) {
                if (srch_step_x * srch_step_y > 0)
                    y_crt_idx += srch_step_y;
                else // srch_step_x * srch_step_y < 0
                    x_crt_idx += srch_step_x;
            }
            else if (srch_drct[0] * heading_drct[1] - srch_drct[1] * heading_drct[0] < 0) {
                if (srch_step_x * srch_step_y > 0)
                    x_crt_idx += srch_step_x;
                else // srch_step_x * srch_step_y < 0
                    y_crt_idx += srch_step_y;
            }
            else { // srch_drct[0] * heading_drct[1] - srch_drct[1] * heading_drct[0] == 0
                if (srch_step_x * heading_drct[0] > srch_step_y * heading_drct[1])
                    x_crt_idx += srch_step_x;
                else if (srch_step_x * heading_drct[0] < srch_step_y * heading_drct[1])
                    y_crt_idx += srch_step_y;
                else { // srch_step_x * heading_drct[0] == srch_step_y * heading_drct[1]
                    x_crt_idx += srch_step_x;
                    y_crt_idx += srch_step_y;
                };
            }
        }
        return false;
    }

    bool is_blocked_dynobj(std::vector<dyn_obj> const& obs_dyn_obj, std::array<double, 2> const& obs_posi, std::array<double, 2> const& cmr_posi)
    {
        std::array<double, 2> obs_vec{obs_posi[0] - cmr_posi[0], obs_posi[1] - cmr_posi[1]};
        for (auto itr = obs_dyn_obj.begin(); itr != obs_dyn_obj.end(); itr++) {
            std::array<double, 2> obj_vec{itr->posi[0] - cmr_posi[0], itr->posi[1] - cmr_posi[1]};
            if (obs_vec[0]*obj_vec[0] + obs_vec[1]*obj_vec[1] > obj_vec[0]*obj_vec[0] + obj_vec[1]*obj_vec[1]) {
                if (((obs_vec[0]*obj_vec[0] + obs_vec[1]*obj_vec[1]) * (obs_vec[0]*obj_vec[0] + obs_vec[1]*obj_vec[1])) / ((obs_vec[0]*obs_vec[0] + obs_vec[1]*obs_vec[1]) * (obj_vec[0]*obj_vec[0] + obj_vec[1]*obj_vec[1])) > ((obj_vec[0]*obj_vec[0] + obj_vec[1]*obj_vec[1]) - itr->radius*itr->radius) / ((obj_vec[0]*obj_vec[0] + obj_vec[1]*obj_vec[1]))) {
                    return true;
                }
            }
        }
        return false;
    }

    double gaussian_distribution(std::array<double, 2> avr, double stdev, std::array<double, 2> pt)
    {
        double dist_sqr = (pt[0]-avr[0]) * (pt[0]-avr[0]) + (pt[1]-avr[1]) * (pt[1]-avr[1]);
        double retval = 1.0 / (2.0 * M_PI * stdev*stdev) * std::exp(-dist_sqr / (2.0 * stdev*stdev));

        return retval;
    }

    bool is_covered_sector(std::array<double, 2> center, double bias_angle, double fov_angle, double fov_depth, std::array<double, 2> pt)
    {
        std::array<double, 2> vec_pt{pt[0] - center[0], pt[1] - center[1]};
        std::array<double, 2> vec_right{std::cos(bias_angle - fov_angle/2), std::sin(bias_angle - fov_angle/2)};
        std::array<double, 2> vec_left{std::cos(bias_angle + fov_angle/2), std::sin(bias_angle + fov_angle/2)};
        if (vec_pt[0]*vec_pt[0] + vec_pt[1]*vec_pt[1] <= fov_depth*fov_depth) {
            if (vec_right[0]*vec_pt[1] - vec_right[1]*vec_pt[0] >= 0 && vec_left[0]*vec_pt[1] - vec_left[1]*vec_pt[0] <= 0) {
                return true;
            }
        }
        return false;
    }

    void generate_urgency_trajpt_matrix(cv::Mat *urgency_trajpt, int urgency_radius_dscrt, int time_delay_idx)
    {
        double time_delay = static_cast<double>(time_delay_idx) * step_time;
        for (size_t i = 0; i < urgency_trajpt->rows; i++){
            for (size_t j = 0; j < urgency_trajpt->cols; j++){
                std::array<double, 2> dist_bgn{(static_cast<double>(urgency_radius_dscrt) - static_cast<double>(i)) * grid_size, (static_cast<double>(urgency_radius_dscrt) - static_cast<double>(j)) * grid_size};
                std::array<double, 2> dist_end{dist_bgn[0] + traj_waypt[time_delay_idx + waypt_interval][0] - traj_waypt[time_delay_idx][0], dist_bgn[1] + traj_waypt[time_delay_idx + waypt_interval][1] - traj_waypt[time_delay_idx][1]};
                double dist_sqr = dist_bgn[0] * dist_bgn[0] + dist_bgn[1] * dist_bgn[1];
                double urgency = 1.0 / (2.0 * M_PI * stdev_vel * stdev_vel) * std::exp(-dist_sqr / (2.0 * (stdev_vel * time_delay) * (stdev_vel * time_delay)));
                std::array<double, 2> vel_bgn{dist_bgn[0] / time_delay, dist_bgn[1] / time_delay};
                std::array<double, 2> vel_end{dist_end[0] / (time_delay + waypt_interval * step_time), dist_end[1] / (time_delay + waypt_interval * step_time)};
                urgency *= std::sqrt((vel_end[0] - vel_bgn[0]) * (vel_end[0] - vel_bgn[0]) + (vel_end[1] - vel_bgn[1]) * (vel_end[1] - vel_bgn[1]));
                urgency_trajpt->at<double>(i, j) = urgency;
            }
        }
    }



    void move_map_potential(std::array<double, 3> const& robot_del_position)
    {
        cv::Mat map_potential_alt{map_potential};
        std::array<int, 2> del_pos_dscrt_xy {static_cast<int>(robot_del_position[0]/grid_size), static_cast<int>(robot_del_position[1]/grid_size)};
        cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1.0, 0.0, del_pos_dscrt_xy[0], 0.0, 1.0, del_pos_dscrt_xy[1]);
        cv::warpAffine(map_potential_alt, map_potential, trans_mat, map_potential_alt.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(prior_prob));
    }

    void scan_map_potential(double camera_last_orientation, double fov_angle, double fov_depth, std::vector<dyn_obj> const& last_obs_dyn_obj, cv::Mat const& last_obs_stat_obj, std::array<double, 3> robot_last_position)
    {
        cv::Point center(map_size_hf_dscrt, map_size_hf_dscrt);
        int fov_depth_dscrt = static_cast<int>(fov_depth / grid_size);
        double start_angle = 0.0;
        double end_angle = fov_angle * 180.0/M_PI;
        double bias_angle = M_PI/2.0 - camera_last_orientation - fov_angle/2.0;
        bias_angle = theta_property(bias_angle);
        if (bias_angle < 0.0)
            bias_angle += 2 * M_PI;
        bias_angle *= 180.0/M_PI;
        
        cv::Mat mask = cv::Mat::ones(map_size_dscrt, map_size_dscrt, CV_64F); // CV_8UC1 is inconvinient for calculation, but it is more efficient than CV_64F
        cv::ellipse(mask, center, cv::Size(fov_depth_dscrt, fov_depth_dscrt), bias_angle, start_angle, end_angle, cv::Scalar(0), -1);

        std::array<int, 2> cmr_pt_idx{map_size_hf_dscrt, map_size_hf_dscrt};
        for (size_t i = 0; i < mask.rows; i++){
            for (size_t j = 0; j < mask.cols; j++){
                if (mask.at<uchar>(i, j) == 1){
                    std::array<int, 2> obs_pt_idx{static_cast<int>(i), static_cast<int>(j)};
                    std::array<double, 2> obs_posi{static_cast<double>(i - map_size_hf_dscrt)*grid_size + robot_last_position[0], static_cast<double>(j - map_size_hf_dscrt)*grid_size + robot_last_position[1]};
                    std::array<double, 2> cmr_posi{robot_last_position[0], robot_last_position[1]};
                    if (is_blocked_grid(last_obs_stat_obj, obs_pt_idx, cmr_pt_idx) || is_blocked_dynobj(last_obs_dyn_obj, obs_posi, cmr_posi)) {
                        mask.at<uchar>(i, j) = 0;
                    }
                }
            }
        }

        map_potential = map_potential.mul(mask);
    }

    void convolute_map_potential()
    {
        cv::Mat map_potential_alt{map_potential};
        cv::filter2D(map_potential_alt, map_potential, -1, cvlt_kern, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    }

    void hstr_update_known_obj()
    {
        for (auto itr = known_obj.begin(); itr != known_obj.end(); itr++) {
            itr->posi[0] += itr->vel[0] * step_time;
            itr->posi[1] += itr->vel[1] * step_time;
            itr->frame_idx++;
        }
    }

    void obs_update_known_obj(std::vector<dyn_obj> const& last_obs_dyn_obj)
    {
        /*/ assignment problem - hungarian algorithm
        for (auto itr_1 = last_obs_dyn_obj.begin(); itr_1 != last_obs_dyn_obj.end(); itr_1++) {
            for (auto itr_2 = known_obj.begin(); itr_2 != known_obj.end(); itr_2++) {
                
            }
        }
        //*/

        for (auto itr = known_obj.begin(); itr != known_obj.end(); itr++) {
            std::array<double, 2> sector_center {robot_last_position[0], robot_last_position[1]};
            std::array<double, 2> pt{itr->posi[0], itr->posi[1]};
            if (is_covered_sector(sector_center, camera_last_orientation, fov_angle, fov_depth, pt)) {
                known_obj.erase(itr);
                itr--;
            } else if ((pt[0] - robot_last_position[0]) * (pt[0] - robot_last_position[0]) + (pt[1] - robot_last_position[1]) * (pt[1] - robot_last_position[1]) > dyn_obj_vanish_dist * dyn_obj_vanish_dist) {
                known_obj.erase(itr);
                itr--;
            }
        }

        for (auto itr = last_obs_dyn_obj.begin(); itr != last_obs_dyn_obj.end(); itr++) {
            known_obj.emplace_back(dyn_obj{{itr->posi[0], itr->posi[1]}, {itr->vel[0], itr->vel[1]}, itr->radius, 0});
        }
    }

    void update_urgency_map_potential()
    {
        urgency_map_potential = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_64F);

        for (size_t i = static_cast<size_t>(waypt_start_idx); i + waypt_interval < waypt_num; i += static_cast<size_t>(waypt_interval)) {
            std::array<int, 2> waypt_idx{static_cast<int>((traj_waypt[i][0] - robot_last_position[0]) / grid_size) + map_size_hf_dscrt, static_cast<int>((traj_waypt[i][1] - robot_last_position[1]) / grid_size) + map_size_hf_dscrt};
            double urgency_radius = 3 * stdev_vel * i * step_time;
            int urgency_radius_dscrt = static_cast<int>(urgency_radius / grid_size);

            cv::Mat urgency_trajpt = pre_calculated_urgency_trajpt[(i - waypt_start_idx) / waypt_interval];

            int board_lim_left = std::min(waypt_idx[0], urgency_radius_dscrt);
            int board_lim_right = std::min(map_size_dscrt - waypt_idx[0], urgency_radius_dscrt+1);
            int board_lim_down = std::min(waypt_idx[1], urgency_radius_dscrt);
            int board_lim_up = std::min(map_size_dscrt - waypt_idx[1], urgency_radius_dscrt+1);
            
            cv::Mat sub_urgency_trajpt = urgency_trajpt(cv::Rect(urgency_radius_dscrt - board_lim_left, urgency_radius_dscrt - board_lim_down, board_lim_right + board_lim_left, board_lim_up + board_lim_down));
            urgency_map_potential(cv::Rect(waypt_idx[0] - board_lim_left, waypt_idx[1] - board_lim_down, board_lim_right + board_lim_left, board_lim_up + board_lim_down)) += sub_urgency_trajpt;
        }

        urgency_map_potential = urgency_map_potential.mul(map_potential);
    }

    void update_urgency_known_obj()
    {
        urgency_known_obj.resize(0);
        for (auto itr = known_obj.begin(); itr != known_obj.end(); itr++) {
            double traj_urgency = 0.0;
            for (size_t i = static_cast<size_t>(waypt_start_idx); i + waypt_interval < waypt_num; i += static_cast<size_t>(waypt_interval)) {
                double del_t_obj_bgn = static_cast<double>(i + 1 + itr->frame_idx) * step_time;
                double del_t_obj_end = static_cast<double>(i + 1 + waypt_interval + itr->frame_idx) * step_time;
                std::array<double, 2> dist_bgn{traj_waypt[i][0] - (itr->posi[0] + itr->vel[0] * del_t_obj_bgn), traj_waypt[i][1] - (itr->posi[1] + itr->vel[1] * del_t_obj_bgn)};
                std::array<double, 2> dist_end{traj_waypt[i + waypt_interval][0] - (itr->posi[0] + itr->vel[0] * del_t_obj_end), traj_waypt[i + waypt_interval][1] - (itr->posi[1] + itr->vel[1] * del_t_obj_end)};
                double dist_sqr = dist_bgn[0] * dist_bgn[0] + dist_bgn[1] * dist_bgn[1];
                double urgency = 1.0 / (2.0 * M_PI * stdev_acc * stdev_acc) * std::exp(-dist_sqr / (2.0 * (1.0/2.0 * stdev_acc * del_t_obj_bgn * del_t_obj_bgn) * (1.0/2.0 * stdev_acc * del_t_obj_bgn * del_t_obj_bgn)));
                std::array<double, 2> acc_bgn{2*dist_bgn[0] / (del_t_obj_bgn * del_t_obj_bgn), 2*dist_bgn[1] / (del_t_obj_bgn * del_t_obj_bgn)};
                std::array<double, 2> acc_end{2*dist_end[0] / (del_t_obj_end * del_t_obj_end), 2*dist_end[1] / (del_t_obj_end * del_t_obj_end)};
                urgency *= std::sqrt((acc_end[0] - acc_bgn[0]) * (acc_end[0] - acc_bgn[0]) + (acc_end[1] - acc_bgn[1]) * (acc_end[1] - acc_bgn[1]));
                traj_urgency += urgency;
            }
            urgency_known_obj.push_back(traj_urgency);
        }
    }

    double get_hybrid_urgency(double camera_new_orientation, std::array<double, 3> const& robot_last_position, std::array<double, 3> const& robot_new_position, double fov_angle, double fov_depth, double fov_depth_min)
    {
        // calculate urgency potential
        std::array<int, 2> del_pos_dscrt_xy{static_cast<int>((robot_new_position[0] - robot_last_position[0]) / grid_size), static_cast<int>((robot_new_position[1] - robot_last_position[1]) / grid_size)};
        cv::Point center(map_size_hf_dscrt + del_pos_dscrt_xy[0], map_size_hf_dscrt + del_pos_dscrt_xy[1]);
        int fov_depth_dscrt = static_cast<int>(fov_depth / grid_size);
        int fov_depth_min_dscrt = static_cast<int>(fov_depth_min / grid_size);
        double start_angle = 0.0;
        double end_angle = (2*M_PI - fov_angle) * 180.0/M_PI;
        double bias_angle = M_PI/2.0 - (camera_new_orientation - M_PI) - (2*M_PI - fov_angle)/2.0;
        bias_angle = theta_property(bias_angle);
        if (bias_angle < 0.0)
            bias_angle += 2 * M_PI;
        bias_angle *= 180.0/M_PI;

        cv::Mat mask = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_64F);
        cv::ellipse(mask, center, cv::Size(fov_depth_dscrt, fov_depth_dscrt), bias_angle, start_angle, end_angle, cv::Scalar(1), -1);
        for (size_t i = 0; i < mask.rows; i++) {
            for (size_t j = 0; j < mask.cols; j++) {
                if (mask.at<double>(i, j) > 0) {
                    double pt_angle = std::atan2(static_cast<double>(j - map_size_hf_dscrt - del_pos_dscrt_xy[1]), static_cast<double>(i - map_size_hf_dscrt - del_pos_dscrt_xy[0]));
                    double pt_rel_angle = theta_property(pt_angle - camera_new_orientation);
                    mask.at<double>(i, j) = hybrid_urgency_grad * (M_PI - std::abs(pt_rel_angle));
                }
            }
        }

        start_angle = 0.0;
        end_angle = fov_angle * 180.0/M_PI;
        bias_angle = M_PI/2.0 - camera_new_orientation - fov_angle/2.0;
        bias_angle = theta_property(bias_angle);
        if (bias_angle < 0.0)
            bias_angle += 2 * M_PI;
        bias_angle *= 180.0/M_PI;
        cv::ellipse(mask, center, cv::Size(fov_depth_dscrt, fov_depth_dscrt), bias_angle, start_angle, end_angle, cv::Scalar(1), -1);

        cv::circle(mask, center, fov_depth_min_dscrt, cv::Scalar(0), -1);

        cv::Mat masked_urgency_map_potential = urgency_map_potential.mul(mask);
        double urgency_potential = cv::sum(masked_urgency_map_potential)[0];

        // calculate urgency known object
        double urgency_known_obj = 0.0;
        int known_obj_idx = 0;
        for (auto itr = known_obj.begin(); itr != known_obj.end(); itr++) {
            double ang_coeff = 0.0;
            std::array<double, 2> obj_rel_posi{itr->posi[0] + itr->vel[0] * step_time - robot_new_position[0], itr->posi[1] + itr->vel[1] * step_time - robot_new_position[1]};
            if (obj_rel_posi[0]*obj_rel_posi[0] + obj_rel_posi[1]*obj_rel_posi[1] <= fov_depth*fov_depth) {
                double obj_angle = std::atan2(obj_rel_posi[1], obj_rel_posi[0]);
                double obj_rel_angle = theta_property(obj_rel_angle - camera_new_orientation);
                if (std::abs(obj_rel_angle) <= fov_angle/2.0) {
                    ang_coeff = 1.0
                }
                else { // std::abs(obj_rel_angle) > fov_angle/2.0
                    ang_coeff = hybrid_urgency_grad * (M_PI - std::abs(obj_rel_angle));
                }
            }
            urgency_known_obj += ang_coeff * urgency_known_obj[known_obj_idx];
        }

        return urgency_potential + urgency_known_obj;
    }



    void update_map()
    {
        std::array<double, 3> robot_last_position_alt = robot_last_position;
        robot_last_position = robot_crt_position;
        std::array<double, 3> robot_del_position{robot_last_position[0] - robot_last_position_alt[0], robot_last_position[1] - robot_last_position_alt[1], robot_last_position[2] - robot_last_position_alt[2]};
        
        camera_last_orientation = camera_crt_orientation;
        last_obs_dyn_obj = crt_obs_dyn_obj;
        last_obs_stat_obj = crt_obs_stat_obj;


        std::chrono::high_resolution_clock::time_point t_0 = std::chrono::high_resolution_clock::now();
        move_map_potential(robot_del_position);
        std::chrono::high_resolution_clock::time_point t_1 = std::chrono::high_resolution_clock::now();
        scan_map_potential(camera_last_orientation, fov_angle, fov_depth, last_obs_dyn_obj, last_obs_stat_obj, robot_last_position);
        std::chrono::high_resolution_clock::time_point t_2 = std::chrono::high_resolution_clock::now();
        convolute_map_potential();
        std::chrono::high_resolution_clock::time_point t_3 = std::chrono::high_resolution_clock::now();
        hstr_update_known_obj();
        std::chrono::high_resolution_clock::time_point t_4 = std::chrono::high_resolution_clock::now();
        obs_update_known_obj(last_obs_dyn_obj);
        std::chrono::high_resolution_clock::time_point t_5 = std::chrono::high_resolution_clock::now();



        std::chrono::high_resolution_clock::time_point t_6 = std::chrono::high_resolution_clock::now();
        update_urgency_map_potential();
        std::chrono::high_resolution_clock::time_point t_7 = std::chrono::high_resolution_clock::now();
        update_urgency_known_obj();
        std::chrono::high_resolution_clock::time_point t_8 = std::chrono::high_resolution_clock::now();
        get_hybrid_urgency(camera_last_orientation, robot_last_position, traj_waypt[0], fov_angle, fov_depth, fov_depth_min);
        std::chrono::high_resolution_clock::time_point t_9 = std::chrono::high_resolution_clock::now();
       
        std::cout << "move_map_potential: " << (t_1 - t_0).count()/1e6 << "ms" << std::endl;
        std::cout << "scan_map_potential: " << (t_2 - t_1).count()/1e6 << "ms" << std::endl;
        std::cout << "convolute_map_potential: " << (t_3 - t_2).count()/1e6 << "ms" << std::endl;
        std::cout << "hstr_update_known_obj: " << (t_4 - t_3).count()/1e6 << "ms" << std::endl;
        std::cout << "obs_update_known_obj: " << (t_5 - t_4).count()/1e6 << "ms" << std::endl;
        std::cout << "update_urgency_map_potential: " << (t_7 - t_6).count()/1e6 << "ms" << std::endl;
        std::cout << "update_urgency_known_obj: " << (t_8 - t_7).count()/1e6 << "ms" << std::endl;
        std::cout << "get_hybrid_urgency: " << (t_9 - t_8).count()/1e6 << "ms" << std::endl;
    }



    //*/
    void draw_map()
    {
        cv::Mat map_potential_img = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_8UC1);
        cv::Mat urgency_map_potential_img = cv::Mat::zeros(map_size_dscrt, map_size_dscrt, CV_64F);
        for (size_t i = 0; i < map_size_dscrt; i++){
            for (size_t j = 0; j < map_size_dscrt; j++){
                map_potential_img.at<uchar>(i, j) = static_cast<uchar>(map_potential.at<double>(i, j) * 255);
            }
        }
        cv::normalize(urgency_map_potential, urgency_map_potential_img, 0, 255, cv::NORM_MINMAX);
        urgency_map_potential_img.convertTo(urgency_map_potential_img, CV_8UC1);
        cv::imshow("map_potential", map_potential_img);
        cv::imshow("urgency_map_potential", urgency_map_potential_img);
        cv::waitKey(0);
    }
    //*/
    
};

#endif