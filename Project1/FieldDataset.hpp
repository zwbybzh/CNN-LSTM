#pragma once
#include <iostream>
#include<cmath>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>

class FieldDataset : public torch::data::Dataset<FieldDataset> {
private:
    int num_samples_;//鸟用没有
    int width_;
    int height_;
    std::mt19937 rng_;
    bool is_rd;
    int example_num;

public:
    FieldDataset(int num_samples=1, int width=8, int height=8,bool is_rd = true ,int example_num = 0)
        : num_samples_(num_samples), width_(width), height_(height),is_rd(is_rd),example_num(example_num) {
        std::random_device rd;
        rng_.seed(rd());
    }

    // 修正get函数签名
    torch::data::Example<> get(size_t index) override {
        // 随机选择场类型
        std::uniform_int_distribution<int> dist_type(0, 2);
        int field_type = 0;
        if (is_rd == true) {
            field_type = dist_type(rng_);
        }
        else{ 
            field_type  = example_num;
        }
        // 创建网格坐标
        std::vector<float> x(width_), y(height_);
        for (int i = 0; i < width_; ++i) {
            x[i] = -1.0f + 2.0f * static_cast<float>(i) / (width_ - 1);
        }
        for (int i = 0; i < height_; ++i) {
            y[i] = -1.0f + 2.0f * static_cast<float>(i) / (height_ - 1);
        }

        // 初始化场向量
        torch::Tensor field_x = torch::zeros({ height_, width_ });
        torch::Tensor field_y = torch::zeros({ height_, width_ });

        if (field_type == 0) { // 均匀场
            std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
            float angle = dist_angle(rng_);
            field_x.fill_(std::cos(angle));
            field_y.fill_(std::sin(angle));
        }
        else if (field_type == 1) { // 旋转场 - 修复版本
            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float dx = x[j];
                    float dy = y[i];
                    float radius = std::sqrt(dx * dx + dy * dy);

                    // 更安全的半径处理
                    radius = std::max(radius, 0.3f);  // 增加最小半径

                    // 直接计算单位向量，避免除法
                    float base_angle = std::atan2(dy, dx);

                    // 旋转90度（顺时针或逆时针）
                    std::uniform_real_distribution<float> dist_dir(0, 1);
                    float rotation_direction = (dist_dir(rng_) > 0.5f) ? 1.0f : -1.0f;
                    float angle = base_angle + rotation_direction * M_PI / 2.0f;

                    // 直接使用单位向量，不除以半径
                    field_x[i][j] = std::cos(angle);
                    field_y[i][j] = std::sin(angle);
                }
            }
        }
        //else if (field_type == 1) { // 旋转场
        //    for (int i = 0; i < height_; ++i) {
        //        for (int j = 0; j < width_; ++j) {
        //            float dx = x[j];
        //            float dy = y[i];
        //            float radius = std::sqrt(dx * dx + dy * dy);
        //            if (radius < 0.1f) radius = 0.1f;
        //            float angle = std::atan2(dy, dx) + M_PI / 2.0f;
        //            // 随机选择旋转方向
        //            std::uniform_real_distribution<float> dist_dir(0, 1);
        //            if (dist_dir(rng_) > 0.5f) {
        //                angle = -angle;
        //            }
        //            field_x[i][j] = std::cos(angle) / radius;
        //            field_y[i][j] = std::sin(angle) / radius;
        //        }
        //    }
        //    // 归一化
        //    auto magnitude = torch::sqrt(field_x.square() + field_y.square());
        //    field_x = field_x / (magnitude + 1e-8f);
        //    field_y = field_y / (magnitude + 1e-8f);
        //}
        else { // 梯度场 - 修复版本
            std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
            float angle_start = dist_angle(rng_);
            float angle_end = fmod(angle_start + M_PI / 2.0f, 2 * M_PI);

            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float weight = (x[j] + 1.0f) / 2.0f;
                    weight = std::clamp(weight, 0.0f, 1.0f);  // 确保权重在合理范围

                    // 线性插值
                    float interp_x = std::cos(angle_start) * (1 - weight) + std::cos(angle_end) * weight;
                    float interp_y = std::sin(angle_start) * (1 - weight) + std::sin(angle_end) * weight;

                    // 归一化
                    float norm = std::sqrt(interp_x * interp_x + interp_y * interp_y);
                    if (norm < 1e-6f) {
                        // 如果模长太小，使用默认方向
                        field_x[i][j] = 1.0f;
                        field_y[i][j] = 0.0f;
                    }
                    else {
                        field_x[i][j] = interp_x / norm;
                        field_y[i][j] = interp_y / norm;
                    }
                }
            }
        }
        //else { // 梯度场
        //    std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
        //    float angle_start = dist_angle(rng_);
        //    float angle_end = fmod(angle_start + M_PI / 2.0f, 2 * M_PI);
        //    for (int i = 0; i < height_; ++i) {
        //        for (int j = 0; j < width_; ++j) {
        //            float weight = (x[j] + 1.0f) / 2.0f;
        //            field_x[i][j] = std::cos(angle_start) * (1 - weight) + std::cos(angle_end) * weight;
        //            field_y[i][j] = std::sin(angle_start) * (1 - weight) + std::sin(angle_end) * weight;
        //        }
        //    }
        //    // 归一化
        //    auto magnitude = torch::sqrt(field_x.square() + field_y.square());
        //    field_x = field_x / (magnitude + 1e-8f);
        //    field_y = field_y / (magnitude + 1e-8f);
        //}

        // 最终的安全检查
        auto final_check = [](torch::Tensor& field_component) {
            auto nan_mask = field_component.isnan();
            auto inf_mask = field_component.isinf();
            if (nan_mask.any().item<bool>() || inf_mask.any().item<bool>()) {
                std::cout << "WARNING: Replacing invalid values in field" << std::endl;
                field_component = torch::where(nan_mask, torch::zeros_like(field_component), field_component);
                field_component = torch::where(inf_mask, torch::zeros_like(field_component), field_component);
            }

            // 限制数值范围
            field_component = torch::clamp(field_component, -1.0f, 1.0f);
            };

        final_check(field_x);
        final_check(field_y);

        // 组合场向量并调整形状为[height, width.2]
        auto field = torch::stack({ field_x, field_y }, 2).to(torch::kFloat32);

        // 验证最终输出
        auto field_min = field.min().item<float>();
        auto field_max = field.max().item<float>();
        if (field_min < -1.1f || field_max > 1.1f) {
            std::cout << "WARNING: Field values out of expected range: "
                << field_min << " to " << field_max << std::endl;
        }

        // 固定起点坐标
        auto start_pos = torch::tensor({ 0.5f, 0.5f }, torch::kFloat32);

        return { field, start_pos };//[w * h*2]
    }

    torch::optional<size_t> size() const override {
        return num_samples_;
    }
};


