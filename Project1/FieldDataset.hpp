#pragma once
#include <iostream>
#include<cmath>
#include <vector>
#include <random>
#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>

class FieldDataset : public torch::data::Dataset<FieldDataset> {
private:
    int num_samples_;//����û��
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

    // ����get����ǩ��
    torch::data::Example<> get(size_t index) override {
        // ���ѡ������
        std::uniform_int_distribution<int> dist_type(0, 2);
        int field_type = 0;
        if (is_rd == true) {
            field_type = dist_type(rng_);
        }
        else{ 
            field_type  = example_num;
        }
        // ������������
        std::vector<float> x(width_), y(height_);
        for (int i = 0; i < width_; ++i) {
            x[i] = -1.0f + 2.0f * static_cast<float>(i) / (width_ - 1);
        }
        for (int i = 0; i < height_; ++i) {
            y[i] = -1.0f + 2.0f * static_cast<float>(i) / (height_ - 1);
        }

        // ��ʼ��������
        torch::Tensor field_x = torch::zeros({ height_, width_ });
        torch::Tensor field_y = torch::zeros({ height_, width_ });

        if (field_type == 0) { // ���ȳ�
            std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
            float angle = dist_angle(rng_);
            field_x.fill_(std::cos(angle));
            field_y.fill_(std::sin(angle));
        }
        else if (field_type == 1) { // ��ת�� - �޸��汾
            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float dx = x[j];
                    float dy = y[i];
                    float radius = std::sqrt(dx * dx + dy * dy);

                    // ����ȫ�İ뾶����
                    radius = std::max(radius, 0.3f);  // ������С�뾶

                    // ֱ�Ӽ��㵥λ�������������
                    float base_angle = std::atan2(dy, dx);

                    // ��ת90�ȣ�˳ʱ�����ʱ�룩
                    std::uniform_real_distribution<float> dist_dir(0, 1);
                    float rotation_direction = (dist_dir(rng_) > 0.5f) ? 1.0f : -1.0f;
                    float angle = base_angle + rotation_direction * M_PI / 2.0f;

                    // ֱ��ʹ�õ�λ�����������԰뾶
                    field_x[i][j] = std::cos(angle);
                    field_y[i][j] = std::sin(angle);
                }
            }
        }
        //else if (field_type == 1) { // ��ת��
        //    for (int i = 0; i < height_; ++i) {
        //        for (int j = 0; j < width_; ++j) {
        //            float dx = x[j];
        //            float dy = y[i];
        //            float radius = std::sqrt(dx * dx + dy * dy);
        //            if (radius < 0.1f) radius = 0.1f;
        //            float angle = std::atan2(dy, dx) + M_PI / 2.0f;
        //            // ���ѡ����ת����
        //            std::uniform_real_distribution<float> dist_dir(0, 1);
        //            if (dist_dir(rng_) > 0.5f) {
        //                angle = -angle;
        //            }
        //            field_x[i][j] = std::cos(angle) / radius;
        //            field_y[i][j] = std::sin(angle) / radius;
        //        }
        //    }
        //    // ��һ��
        //    auto magnitude = torch::sqrt(field_x.square() + field_y.square());
        //    field_x = field_x / (magnitude + 1e-8f);
        //    field_y = field_y / (magnitude + 1e-8f);
        //}
        else { // �ݶȳ� - �޸��汾
            std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
            float angle_start = dist_angle(rng_);
            float angle_end = fmod(angle_start + M_PI / 2.0f, 2 * M_PI);

            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float weight = (x[j] + 1.0f) / 2.0f;
                    weight = std::clamp(weight, 0.0f, 1.0f);  // ȷ��Ȩ���ں���Χ

                    // ���Բ�ֵ
                    float interp_x = std::cos(angle_start) * (1 - weight) + std::cos(angle_end) * weight;
                    float interp_y = std::sin(angle_start) * (1 - weight) + std::sin(angle_end) * weight;

                    // ��һ��
                    float norm = std::sqrt(interp_x * interp_x + interp_y * interp_y);
                    if (norm < 1e-6f) {
                        // ���ģ��̫С��ʹ��Ĭ�Ϸ���
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
        //else { // �ݶȳ�
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
        //    // ��һ��
        //    auto magnitude = torch::sqrt(field_x.square() + field_y.square());
        //    field_x = field_x / (magnitude + 1e-8f);
        //    field_y = field_y / (magnitude + 1e-8f);
        //}

        // ���յİ�ȫ���
        auto final_check = [](torch::Tensor& field_component) {
            auto nan_mask = field_component.isnan();
            auto inf_mask = field_component.isinf();
            if (nan_mask.any().item<bool>() || inf_mask.any().item<bool>()) {
                std::cout << "WARNING: Replacing invalid values in field" << std::endl;
                field_component = torch::where(nan_mask, torch::zeros_like(field_component), field_component);
                field_component = torch::where(inf_mask, torch::zeros_like(field_component), field_component);
            }

            // ������ֵ��Χ
            field_component = torch::clamp(field_component, -1.0f, 1.0f);
            };

        final_check(field_x);
        final_check(field_y);

        // ��ϳ�������������״Ϊ[height, width.2]
        auto field = torch::stack({ field_x, field_y }, 2).to(torch::kFloat32);

        // ��֤�������
        auto field_min = field.min().item<float>();
        auto field_max = field.max().item<float>();
        if (field_min < -1.1f || field_max > 1.1f) {
            std::cout << "WARNING: Field values out of expected range: "
                << field_min << " to " << field_max << std::endl;
        }

        // �̶��������
        auto start_pos = torch::tensor({ 0.5f, 0.5f }, torch::kFloat32);

        return { field, start_pos };//[w * h*2]
    }

    torch::optional<size_t> size() const override {
        return num_samples_;
    }
};


