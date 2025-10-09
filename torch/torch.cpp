#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
//#include <matplot/matplot.h>
//#include"matplotlibcpp.h"

//namespace plt = matplotlibcpp;
// ���ݼ��ࣺ���ɲ�ͬ���͵����߳�
class FieldDataset : public torch::data::Dataset<FieldDataset> {
private:
    int num_samples_;
    int width_;
    int height_;
    std::mt19937 rng_;

public:
    // ���캯��
    FieldDataset(int num_samples, int width, int height)
        : num_samples_(num_samples), width_(width), height_(height) {
        std::random_device rd;
        rng_.seed(rd());
    }

    // ����������
    torch::data::Example<> get(size_t index) override {
        // ���ѡ������
        std::uniform_int_distribution<int> dist_type(0, 2);
        int field_type = dist_type(rng_);

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
        else if (field_type == 1) { // ��ת��
            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float dx = x[j];
                    float dy = y[i];
                    float radius = std::sqrt(dx * dx + dy * dy);
                    if (radius < 0.1f) radius = 0.1f;

                    float angle = std::atan2(dy, dx) + M_PI / 2.0f;

                    // ���ѡ����ת����
                    std::uniform_real_distribution<float> dist_dir(0, 1);
                    if (dist_dir(rng_) > 0.5f) {
                        angle = -angle;
                    }

                    field_x[i][j] = std::cos(angle) / radius;
                    field_y[i][j] = std::sin(angle) / radius;
                }
            }

            // ��һ��
            auto magnitude = torch::sqrt(field_x.square() + field_y.square());
            field_x = field_x / magnitude;
            field_y = field_y / magnitude;
        }
        else { // �ݶȳ�
            std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
            float angle_start = dist_angle(rng_);
            float angle_end = fmod(angle_start + M_PI / 2.0f, 2 * M_PI);

            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float weight = (x[j] + 1.0f) / 2.0f; // �����Ҵ�0���ɵ�1
                    field_x[i][j] = std::cos(angle_start) * (1 - weight) + std::cos(angle_end) * weight;
                    field_y[i][j] = std::sin(angle_start) * (1 - weight) + std::sin(angle_end) * weight;
                }
            }
        }

        // ��ϳ�������������״Ϊ[2, height, width]
        auto field = torch::stack({ field_x, field_y }).to(torch::kFloat32);

        // ���ѡ����㣨��������
        /*std::uniform_real_distribution<float> dist_x(width_ * 0.3f, width_ * 0.7f);
        std::uniform_real_distribution<float> dist_y(height_ * 0.3f, height_ * 0.7f);
        auto start_pos = torch::tensor({ dist_x(rng_), dist_y(rng_) }, torch::kFloat32);*/
        //�̶�1��1
        auto start_pos = torch::tensor({ 0.5f, 0.5f }, torch::kFloat32);
        return { field, start_pos };
    }

    // �������ݼ���С
    torch::optional<size_t> size() const override {
        return num_samples_;
    }
};

// ·���滮ģ�ͣ�CNN + LSTM
//TORCH_MODULE (PathPlannerModel);
class PathPlannerModel : public torch::nn::Module {
private:
    int width_;
    int height_;
    int total_pixels_;
    int hidden_dim_;

    // CNN�㣺��ȡ������
    torch::nn::Sequential cnn{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 64, 3).padding(1)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
        torch::nn::BatchNorm2d(128),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    };

    // LSTM�㣺����·������
    torch::nn::LSTM lstm{
        torch::nn::LSTMOptions(128 + 2 + 32, 128)
            .num_layers(2)
            .batch_first(true)
            .dropout(0.2)
    };

    // ����㣺Ԥ����һ��λ��
    torch::nn::Sequential output_head{
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(64, 2)
    };

    // �ѷ������ص�Ƕ���
    torch::nn::Embedding visited_embedding{
        torch::nn::EmbeddingOptions(1024 + 1, 32) // �����������Ϊ1024
    };

public:
    // ���캯��
    PathPlannerModel(int width, int height, int hidden_dim = 128)
        : width_(width), height_(height), hidden_dim_(hidden_dim) {
        total_pixels_ = width * height;

        // ע��ģ��
        register_module("cnn", cnn);
        register_module("lstm", lstm);
        register_module("output_head", output_head);
        register_module("visited_embedding", visited_embedding);

        // ��ʼ��Ȩ��
        init_weights();
    }

    // ��ʼ��Ȩ��
    void init_weights() {
        for (auto& module : modules(false)) {
            if (auto* conv = module->as<torch::nn::Conv2d>()) {
                torch::nn::init::kaiming_normal_(conv->weight, 0, torch::kFanOut, torch::kReLU);
                if (conv->bias.defined()) {
                    torch::nn::init::constant_(conv->bias, 0);
                }
            }
            else if (auto* bn = module->as<torch::nn::BatchNorm2d>()) {
                torch::nn::init::constant_(bn->weight, 1);
                torch::nn::init::constant_(bn->bias, 0);
            }
            else if (auto* linear = module->as<torch::nn::Linear>()) {
                torch::nn::init::normal_(linear->weight, 0, 0.01);
                torch::nn::init::constant_(linear->bias, 0);
            }
        }
    }

    // ǰ�򴫲�
    std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        forward(
            const torch::Tensor& field_,
            const torch::Tensor& current_pos,
            const torch::Tensor& visited_count,
            std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt
        ) {
        int batch_size = field_.size(0);
        int subpixel_scale = 3;
        //�쳣���
        if (torch::isnan(field_).any().item<bool>()) {
            throw std::runtime_error("NaN detected in field input");
        }
        if (torch::isnan(current_pos).any().item<bool>()) {
            throw std::runtime_error("NaN detected in current_pos input");
        }
        auto field = field_.contiguous();
        // 1. ʹ��CNN��ȡ������
        //std::cout << field.sizes() << std::endl;

        auto field_features = cnn->forward(field); // [batch, hidden_dim, height, width]

        // 2. ��ȡ��ǰλ�õĳ�������˫���Բ�ֵ��
        //std::cout << current_pos << std::endl;

        auto x_norm = (current_pos.index({ torch::indexing::Slice(), 0 }) / (width_ - 1)) * 2 - 1;
        auto y_norm = (current_pos.index({ torch::indexing::Slice(), 1 }) / (height_ - 1)) * 2 - 1;

        auto grid = torch::stack({ x_norm, y_norm }, -1).unsqueeze(1).unsqueeze(1); // [batch, 1,1, 2]

        auto current_features = torch::nn::functional::grid_sample(
            field_features,
            grid,
            torch::nn::functional::GridSampleFuncOptions()

            .mode(torch::kBilinear)//˫���Բ�ֵ�õ�ƽ������
            .padding_mode(torch::kZeros)
            .align_corners(true)//������뵽���ؽ���
        );

        current_features = current_features.squeeze(-1).squeeze(-1); // [batch, hidden_dim]

        //std::cout << current_features.sizes() << std::endl;
        // 3. �����ѷ�����Ϣ
        auto visited_num = (visited_count > 0).to(torch::kFloat32).sum(1).to(torch::kLong);
        auto visited_feat = visited_embedding->forward(visited_num); // [batch, 32]

        // 4. ׼��LSTM����
        auto normalized_pos = current_pos / torch::tensor({ static_cast<float>(width_), static_cast<float>(height_) },
            torch::dtype(torch::kFloat32).device(current_pos.device()));
        auto lstm_input = torch::cat({ current_features, normalized_pos, visited_feat }, 1).unsqueeze(1);

        // 5. LSTMǰ�򴫲�
        auto lstm_out = lstm->forward(lstm_input, hidden_state);

        // 6. Ԥ����һ��λ�õ�ƫ��
        auto delta = output_head->forward(std::get<0>(lstm_out).squeeze(1));

        // �������ƫ��
       delta = torch::tanh(delta) * (2.0f/subpixel_scale);

       // ========== �޸���Сƫ������ ==========
   // ������Сƫ�ƣ��������������������
       float base_min_offset = 0.1f; // ��һ����С����Ϊ�����и����ѡ��

       // �����ƶ������볡�����һ����
       auto grid_x_norm = (current_pos.index({ torch::indexing::Slice(), 0 }) / (width_ - 1)) * 2 - 1;
       auto grid_y_norm = (current_pos.index({ torch::indexing::Slice(), 1 }) / (height_ - 1)) * 2 - 1;
       auto pos_grid = torch::stack({ grid_x_norm, grid_y_norm }, -1).unsqueeze(1).unsqueeze(1);

       // ��ȡ��ǰλ�õĳ�����
       auto current_field = torch::nn::functional::grid_sample(
           field, pos_grid,
           torch::nn::functional::GridSampleFuncOptions()
           .mode(torch::kBilinear)
           .padding_mode(torch::kZeros)
           .align_corners(true)
       ).squeeze(-1).squeeze(-1);

       // ��һ��������
       auto field_norm = torch::norm(current_field, 2, -1, true) + 1e-8f;
       auto field_dir = current_field / field_norm;

       // ��һ���ƶ�����
       auto delta_norm = torch::norm(delta, 2, -1, true) + 1e-8f;
       auto delta_dir = delta / delta_norm;

       // ���㷽��һ���ԣ������
       auto alignment = (field_dir * delta_dir).sum(-1);

       // ��̬������Сƫ��
       auto min_offset = torch::full({ batch_size }, base_min_offset,
           torch::dtype(torch::kFloat32).device(delta.device()));

       // ��ȡ��ǰλ���������������е�����
       auto current_pos_subpixel = current_pos * subpixel_scale;
       auto current_subpixel_x = (current_pos_subpixel.index({ torch::indexing::Slice(), 0 })).clamp(0, width_ * subpixel_scale - 1).to(torch::kLong);
       auto current_subpixel_y = (current_pos_subpixel.index({ torch::indexing::Slice(), 1 })).clamp(0, height_ * subpixel_scale - 1).to(torch::kLong);
       auto current_subpixel_ids = current_subpixel_x * (height_ * subpixel_scale) + current_subpixel_y;

       // �����ܵ�ƫ�Ƶ������ԣ����������ط��ʣ�
       for (int b = 0; b < batch_size; ++b) {
           // ���㵱ǰλ����ԭʼ�����е�����
           auto pixel_x = current_pos.index({ b, 0 }).clamp(0, width_ - 1).to(torch::kLong).item<int>();
           auto pixel_y = current_pos.index({ b, 1 }).clamp(0, height_ - 1).to(torch::kLong).item<int>();
           auto pixel_id = pixel_x * height_ + pixel_y;

           int visit_count = visited_count[b][pixel_id].item<int>();
           float align_score = alignment[b].item<float>();

           // �������Ա��ֲ��䣬����ֵ������Ҫ΢��
           if (align_score > 0.7f) {
               min_offset[b] = base_min_offset * 0.5f;
           }
           else if (visit_count > 5 && align_score < -0.3f) {
               min_offset[b] = base_min_offset * 2.0f;
           }
           else if (visit_count > 3) {
               min_offset[b] = base_min_offset * 1.5f;
           }
       }

       // Ӧ����Сƫ������
       auto delta_magnitude = torch::norm(delta, 2, -1, true);
       auto enforced_magnitude = torch::max(delta_magnitude, min_offset.unsqueeze(-1));
       delta = delta_dir * enforced_magnitude;
       // ========== ��Сƫ�����ƽ��� ==========
      
        // ������һ��λ��
        auto next_pos = current_pos + delta;

        // ��ǿ�߽紦��
        auto near_boundary_x = (next_pos.index({ torch::indexing::Slice(), 0 }) < 1.0f) |
            (next_pos.index({ torch::indexing::Slice(), 0 }) > width_ - 2.0f);
        auto near_boundary_y = (next_pos.index({ torch::indexing::Slice(), 1 }) < 1.0f) |
            (next_pos.index({ torch::indexing::Slice(), 1 }) > height_ - 2.0f);
        auto near_boundary = near_boundary_x | near_boundary_y;

        // ��������߽磬���跴����
        auto boundary_force = torch::zeros_like(delta);
        boundary_force.index({ near_boundary_x, 0 }) = (width_ / 2.0f - next_pos.index({ near_boundary_x, 0 })) * 0.1f;
        boundary_force.index({ near_boundary_y, 1 }) = (height_ / 2.0f - next_pos.index({ near_boundary_y, 1 })) * 0.1f;

        next_pos = next_pos + boundary_force;

        // ȷ��λ��������Χ��
        float margin = 0.5f / subpixel_scale; // �߽�������������ص����
        next_pos = torch::clamp(
            next_pos,
            torch::tensor(margin, torch::device(next_pos.device())),
            torch::tensor({ static_cast<float>(width_ - 1 - margin),
                          static_cast<float>(height_ - 1 - margin) },
                torch::device(next_pos.device()))
        );
        // ========== ���̽������������ѵ��ʱ�� ==========
        if (this->is_training()) {
            // ���㵱ǰλ�õ���������
            auto current_pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width_ - 1).to(torch::kLong);
            auto current_pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height_ - 1).to(torch::kLong);
            auto current_pixel_ids = current_pixel_x * height_ + current_pixel_y;

            // ���ڷ��ʴ���������Ӧ����
            auto visit_counts = visited_count.gather(1, current_pixel_ids.unsqueeze(1)).squeeze(1);
            auto noise_scale = torch::where(
                visit_counts > 10,
                torch::tensor(0.5f, torch::device(next_pos.device())),  // �߷���������������
                torch::tensor(0.1f, torch::device(next_pos.device()))   // �ͷ��������������
            );

            auto exploration_noise = torch::randn_like(next_pos) * noise_scale.unsqueeze(1);
            next_pos = next_pos + exploration_noise;

            // �ٴ�ȷ��λ��������Χ��
            next_pos = torch::clamp(
                next_pos,
                torch::tensor(0.5f, torch::device(next_pos.device())),
                torch::tensor({ static_cast<float>(width_ - 1.5), static_cast<float>(height_ - 1.5) },
                    torch::device(next_pos.device()))
            );
        }
        // ========== ̽���������� ==========

        return { next_pos, std::get<1>(lstm_out) };
    }
};

// �Զ�����ʧ����
class PathLoss {
private:
    float smooth_weight_;
    float align_weight_;
    float coverage_weight_;
    float overlap_weight_;

public:
    // ���캯��
    PathLoss(float smooth_weight = 1.0f, float align_weight = 1.0f,
        float coverage_weight = 10.0f, float overlap_weight = 0.5f)
        : smooth_weight_(smooth_weight), align_weight_(align_weight),
        coverage_weight_(coverage_weight), overlap_weight_(overlap_weight) {
    }

    // ������ʧ
    //std::pair<torch::Tensor, std::unordered_map<std::string, float>>
    //    operator()(const torch::Tensor& path, const torch::Tensor& field, int width, int height) {
    //    int batch_size = path.size(0);
    //    int seq_len = path.size(1);
    //    int total_pixels = width * height;

    //    // 1. ƽ������ʧ
    //    torch::Device device = path.device();
    //    torch::Tensor smooth_loss = torch::tensor(0.0f, device);
    //    if (seq_len > 2) {
    //        auto dirs = path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) }) -
    //            path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) });
    //        auto dir_magnitudes = dirs.norm(2, 2, true) + 1e-8f;
    //        auto dirs_normalized = dirs / dir_magnitudes;

    //        auto dir_changes = dirs_normalized.index({ torch::indexing::Slice(), torch::indexing::Slice(1) }) -
    //            dirs_normalized.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1) });
    //        smooth_loss = dir_changes.norm(2, 2).mean();
    //    }

    //    // 2. ��������ʧ
    //    torch::Tensor align_loss = torch::tensor(0.0f, device = path.device());
    //    if (seq_len > 1) {
    //        //��·�����ڵ���е�
    //        auto mid_points = (path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) }) +
    //            path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) })) / 2.0f;
    //        
    //        auto x_norm = (mid_points.index({ torch::indexing::Slice(), torch::indexing::Slice(), 0 }) / (width - 1)) * 2 - 1;
    //        auto y_norm = (mid_points.index({ torch::indexing::Slice(), torch::indexing::Slice(), 1 }) / (height - 1)) * 2 - 1;
    //        auto grid = torch::stack({ x_norm, y_norm }, -1).unsqueeze(1);

    //        auto sampled_field = torch::nn::functional::grid_sample(
    //            field,
    //            grid,
    //            torch::nn::functional::GridSampleFuncOptions()
    //            .align_corners(true)
    //            .mode(torch::kBilinear)
    //        );

    //        sampled_field = sampled_field.squeeze(2).permute({ 0, 2, 1 });
    //        //�����ǴӴ�·���в�������һ��
    //        // auto sampled_field_normalized = torch::nn::functional::normalize(sampled_field, 2, 2);

    //        auto sampled_field_normalized = torch::nn::functional::normalize(
    //            sampled_field,
    //            torch::nn::functional::NormalizeFuncOptions()
    //            .p(2)          // �����Ľ�������Ӧ��һ����������2��
    //            .dim(2)        // Ҫ�淶����ά�ȣ���Ӧ�ڶ�����������2��
    //        );

    //        //����·���ƶ��ķ���
    //        auto dirs = path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) }) -
    //            path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) });
    //        auto dir_magnitudes = dirs.norm(2, 2, true) + 1e-8f;
    //        auto dirs_normalized = dirs / dir_magnitudes;

    //        auto dot_product = (dirs_normalized * sampled_field).sum(-1);
    //        //����ע�͵���ԭ����������ų�����ͷ��ܶࡣ
    //        /*auto alignment = (dirs_normalized * sampled_field_normalized).sum(2);
    //        align_loss = (1 - alignment).mean();*/
    //        //��sin��ֵ
    //        auto cross_product = torch::abs(
    //            dirs_normalized.index({ "...", 0 }) * sampled_field_normalized.index({ "...", 1 }) -
    //            dirs_normalized.index({ "...", 1 }) * sampled_field_normalized.index({ "...", 0 })
    //        );//�ͷ���ֱ�ƶ�
    //        auto align_loss_m = 1.0 - dot_product.abs();//�ͷ��Ƕ�ƫ��
    //        align_loss = (cross_product+ align_loss_m).mean();  // ֱ����sin�ȵľ�ֵ��Ϊ��ʧ
    //    }

    //    // 3. ���Ƕ���ʧ
    //    //auto pixel_x = path.index({ torch::indexing::Slice(), torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
    //    //auto pixel_y = path.index({ torch::indexing::Slice(), torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
    //    //auto pixel_ids = pixel_x * height + pixel_y;//����Ψһ����id
    //    // 
    //    ////ͳ��ÿһ�����صķ��ʴ���
    //    //torch::Tensor visited = torch::zeros({ batch_size, total_pixels }, device = path.device());
    //    //for (int i = 0; i < seq_len; ++i) {
    //    //    auto ids = pixel_ids.index({ torch::indexing::Slice(), i }).unsqueeze(1);
    //    //    visited.scatter_add_(1, ids, torch::ones({ batch_size, 1 }, device = path.device()));
    //    //}

    //    //auto coverage = (visited > 0).to(torch::kFloat32).mean(1);//���㸲�Ƕ�
    //    //auto coverage_loss = (1 - coverage).mean();

    //    //// 4. �ص��ͷ���ʧ
    //    //auto overlap = (visited - 1).clamp_min(0);
    //    //auto overlap_loss = overlap.mean();
    //     
    //    // 3. ���Ƕ���ʧ - �Ľ��汾������·����Ⱥ��߶θ��ǣ�
    //    
    //   
    //    // ��������ͼ������·�����
    //    float path_width = 1.0f; // ·����ȣ����Ը�����Ҫ����
    //    torch::Tensor visited = torch::zeros({ batch_size, total_pixels }, device = path.device());

    //    for (int b = 0; b < batch_size; ++b) {
    //        for (int i = 0; i < seq_len - 1; ++i) {
    //            // ��ȡ��ǰ�߶������յ�
    //            auto p1 = path.index({ b, i });
    //            auto p2 = path.index({ b, i + 1 });

    //            float x1 = p1[0].item<float>();
    //            float y1 = p1[1].item<float>();
    //            float x2 = p2[0].item<float>();
    //            float y2 = p2[1].item<float>();

    //            // �����߶η�������
    //            float dx = x2 - x1;
    //            float dy = y2 - y1;
    //            float length = std::sqrt(dx * dx + dy * dy);

    //            // ����Ӧ�����������߶γ��Ⱦ�����������
    //            int num_samples = std::max(2, static_cast<int>(length / 0.5f));

    //            for (int s = 0; s < num_samples; ++s) {
    //                float t = static_cast<float>(s) / (num_samples - 1);
    //                float x = x1 + t * dx;
    //                float y = y1 + t * dy;

    //                // ����·����ȣ������Χ����
    //                int px = static_cast<int>(std::round(x));
    //                int py = static_cast<int>(std::round(y));

    //                // ���·����ȷ�Χ�ڵ�����
    //                int half_width = static_cast<int>(std::ceil(path_width / 2));
    //                for (int wx = -half_width; wx <= half_width; ++wx) {
    //                    for (int wy = -half_width; wy <= half_width; ++wy) {
    //                        int current_x = px + wx;
    //                        int current_y = py + wy;

    //                        if (current_x >= 0 && current_x < width && current_y >= 0 && current_y < height) {
    //                            int pixel_id = current_x * height + current_y;
    //                            visited[b][pixel_id] += 1.0;
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //    }

    //    // �Ľ��ĸ��Ƕȼ���
    //    auto coverage_mask = (visited > 0).to(torch::kFloat32);
    //    auto coverage = coverage_mask.view({ batch_size, -1 }).mean(1);
    //    auto coverage_loss = (1 - coverage).mean();

    //    // 4. �Ľ����ص��ͷ���ʧ - ʹ��ƽ���ͷ�����
    //    auto overlap_count = visited - coverage_mask; // ��ȥ�������ǣ���һ�η��ʣ�
    //    auto smooth_overlap = torch::log(1.0 + overlap_count.clamp_min(0)); // ����ƽ����������ȳͷ�
    //    auto overlap_loss = smooth_overlap.mean();

    //    // ����ʧ
    //    auto total_loss = smooth_weight_ * smooth_loss +
    //        align_weight_ * align_loss +
    //        coverage_weight_ * coverage_loss +
    //        overlap_weight_ * overlap_loss;

    //    // ��ʧ���������ڼ�أ�
    //    std::unordered_map<std::string, float> components;
    //    components["total"] = total_loss.item<float>();
    //    components["smooth"] = smooth_loss.item<float>();
    //    components["align"] = align_loss.item<float>();
    //    components["coverage"] = coverage_loss.item<float>();
    //    components["overlap"] = overlap_loss.item<float>();

    //    return { total_loss, components };
    //}
std::pair<torch::Tensor, std::unordered_map<std::string, float>>
operator()(const torch::Tensor& path, const torch::Tensor& field, int width, int height) {
    int batch_size = path.size(0);
    int seq_len = path.size(1);

    // �������������
    int subpixel_scale = 3; // ÿ�����ػ���Ϊ3x3������
    int subpixel_width = width * subpixel_scale;
    int subpixel_height = height * subpixel_scale;
    int total_subpixels = subpixel_width * subpixel_height;

    torch::Device device = path.device();

    // 1. ƽ������ʧ�����ֲ��䣩
    torch::Tensor smooth_loss = torch::tensor(0.0f, device);
    if (seq_len > 2) {
        auto dirs = path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) }) -
            path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) });
        auto dir_magnitudes = dirs.norm(2, 2, true) + 1e-8f;
        auto dirs_normalized = dirs / dir_magnitudes;

        auto dir_changes = dirs_normalized.index({ torch::indexing::Slice(), torch::indexing::Slice(1) }) -
            dirs_normalized.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1) });
        smooth_loss = dir_changes.norm(2, 2).mean();
    }

    // 2. ��������ʧ�����ֲ��䣩
    torch::Tensor align_loss = torch::tensor(0.0f, device = path.device());
    if (seq_len > 1) {
        auto mid_points = (path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) }) +
            path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) })) / 2.0f;

        auto x_norm = (mid_points.index({ torch::indexing::Slice(), torch::indexing::Slice(), 0 }) / (width - 1)) * 2 - 1;
        auto y_norm = (mid_points.index({ torch::indexing::Slice(), torch::indexing::Slice(), 1 }) / (height - 1)) * 2 - 1;
        auto grid = torch::stack({ x_norm, y_norm }, -1).unsqueeze(1);

        auto sampled_field = torch::nn::functional::grid_sample(
            field,
            grid,
            torch::nn::functional::GridSampleFuncOptions()
            .align_corners(true)
            .mode(torch::kBilinear)
        );

        sampled_field = sampled_field.squeeze(2).permute({ 0, 2, 1 });

        auto sampled_field_normalized = torch::nn::functional::normalize(
            sampled_field,
            torch::nn::functional::NormalizeFuncOptions()
            .p(2)
            .dim(2)
        );

        auto dirs = path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) }) -
            path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) });
        auto dir_magnitudes = dirs.norm(2, 2, true) + 1e-8f;
        auto dirs_normalized = dirs / dir_magnitudes;

        auto dot_product = (dirs_normalized * sampled_field).sum(-1);
        auto cross_product = torch::abs(
            dirs_normalized.index({ "...", 0 }) * sampled_field_normalized.index({ "...", 1 }) -
            dirs_normalized.index({ "...", 1 }) * sampled_field_normalized.index({ "...", 0 })
        );
        auto align_loss_m = 1.0 - dot_product.abs();
        align_loss = (cross_product + align_loss_m).mean();
    }

    // 3. �µĸ��Ƕ���ʧ����������������
    torch::Tensor visited_subpixels = torch::zeros({ batch_size, total_subpixels }, device = device);
    torch::Tensor coverage_loss = torch::tensor(0.0f, device);
    torch::Tensor overlap_loss = torch::tensor(0.0f, device);
    torch::Tensor repetition_penalty = torch::tensor(0.0f, device);

    if (seq_len > 0) {
        // ��·����ӳ�䵽����������
        auto path_subpixel = path * subpixel_scale; // ����·���㵽����������

        // ��¼ÿ����ѡ���������ID�������ظ��ͷ���
        torch::Tensor selected_subpixel_ids = torch::zeros({ batch_size, seq_len },
            torch::dtype(torch::kLong).device(device));

        for (int b = 0; b < batch_size; ++b) {
            std::vector<int> visited_this_batch; // ��¼����batch�з��ʹ���������

            for (int i = 0; i < seq_len; ++i) {
                // ��ȡ��ǰ·�����������������е�����
                float x = path_subpixel[b][i][0].item<float>();
                float y = path_subpixel[b][i][1].item<float>();

                // �ҵ���������������ĵ㣨9����ѡ��֮һ��
                int base_x = static_cast<int>(std::floor(x / subpixel_scale)) * subpixel_scale;
                int base_y = static_cast<int>(std::floor(y / subpixel_scale)) * subpixel_scale;

                // 9����ѡ������ƫ��
                std::vector<std::pair<int, int>> candidate_offsets = {
                    {1, 1}, {1, 0}, {1, 2},  // �м���
                    {0, 1}, {0, 0}, {0, 2},  // ����
                    {2, 1}, {2, 0}, {2, 2}   // ����
                };

                // �ҵ���������ĺ�ѡ��
                float min_dist = std::numeric_limits<float>::max();
                int best_candidate = 0;
                int candidate_x = 0, candidate_y = 0;

                for (int c = 0; c < candidate_offsets.size(); ++c) {
                    int cx = base_x + candidate_offsets[c].first;
                    int cy = base_y + candidate_offsets[c].second;
                    float dist = std::sqrt(std::pow(cx - x, 2) + std::pow(cy - y, 2));

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_candidate = c;
                        candidate_x = cx;
                        candidate_y = cy;
                    }
                }

                // ��¼ѡ���������ID
                int subpixel_id = candidate_x * subpixel_height + candidate_y;
                selected_subpixel_ids[b][i] = subpixel_id;

                // ��ǵ�ǰ�����ؼ���8����Ϊ�Ѹ���
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        int nx = candidate_x + dx;
                        int ny = candidate_y + dy;

                        if (nx >= 0 && nx < subpixel_width && ny >= 0 && ny < subpixel_height) {
                            int neighbor_id = nx * subpixel_height + ny;
                            visited_subpixels[b][neighbor_id] += 1.0;
                        }
                    }
                }

                // ����Ƿ��ظ�ѡ��Ӳ�Թ涨�ͷ���
                if (std::find(visited_this_batch.begin(), visited_this_batch.end(), subpixel_id) !=
                    visited_this_batch.end()) {
                    // �ظ�ѡ�񣬸����سͷ�
                    repetition_penalty = repetition_penalty + 10.0f;
                }
                visited_this_batch.push_back(subpixel_id);
            }
        }

        // ���㸲�Ƕ���ʧ
        auto coverage_mask = (visited_subpixels > 0).to(torch::kFloat32);
        auto coverage = coverage_mask.view({ batch_size, -1 }).mean(1);
        coverage_loss = (1 - coverage).mean();

        // �����ص���ʧ��ֻ����8������ص������������ĵ㱾��
        auto center_visited = torch::zeros({ batch_size, total_subpixels }, device = device);

        // ֻ������ĵ㣨��������8�����ص���
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < seq_len; ++i) {
                int subpixel_id = selected_subpixel_ids[b][i].item<int>();
                center_visited[b][subpixel_id] = 1.0;
            }
        }

        // �ص� = �ܸ��� - ���ĵ㸲�ǣ���8������ص����֣�
        auto overlap = visited_subpixels - center_visited;
        auto valid_overlap = torch::max(overlap - 1.0, torch::tensor(0.0f)); // ��ȥ�����ص�
        overlap_loss = torch::log(1.0 + valid_overlap).mean();
    }

    // 4. ̽����������������δ̽������
    torch::Tensor exploration_reward = torch::tensor(0.0f, device);
    if (seq_len > 5) {
        // ���ڸ��Ƕȵ�̽������
        auto coverage_mask = (visited_subpixels > 0).to(torch::kFloat32);
        auto coverage = coverage_mask.view({ batch_size, -1 }).mean(1);
        exploration_reward = -coverage.mean() * 0.1f; // ���Ƕ�Խ�ͣ�����Խ��
    }

    // 5. ͣ�ͼ��Ͷ�̬Ȩ�ص���
    float dynamic_smooth_weight = smooth_weight_;
    float dynamic_align_weight = align_weight_;
    torch::Tensor stagnation_penalty = torch::tensor(0.0f, device);

    if (seq_len > 10) {
        // �����������Ƿ�ͣ�ͣ���ͬһ�������ڣ�
        int check_steps = std::min(5, seq_len / 3);
        auto recent_path = path.index({ torch::indexing::Slice(),
                                      torch::indexing::Slice(seq_len - check_steps, seq_len) });

        // ��·����ӳ�䵽��������
        auto recent_pixels = recent_path / subpixel_scale;
        auto pixel_changes = (recent_pixels.index({ torch::indexing::Slice(),
                                                 torch::indexing::Slice(1, check_steps) }) -
            recent_pixels.index({ torch::indexing::Slice(),
                                 torch::indexing::Slice(0, check_steps - 1) })).norm(2, 2);

        // ����������ƽ���ƶ�С��0.3�����أ���Ϊͣ��
        if (pixel_changes.mean().item<float>() < 0.3f) {
            dynamic_smooth_weight = smooth_weight_ * 0.2f;
            dynamic_align_weight = align_weight_ * 0.2f;
            stagnation_penalty = torch::tensor(5.0f, device);
        }
    }

    // ����ʧ
    auto total_loss = dynamic_smooth_weight * smooth_loss +
        dynamic_align_weight * align_loss +
        coverage_weight_ * coverage_loss +
        overlap_weight_ * overlap_loss +
        repetition_penalty +
        stagnation_penalty +
        exploration_reward;

    // ��ʧ���������ڼ�أ�
    std::unordered_map<std::string, float> components;
    components["total"] = total_loss.item<float>();
    components["smooth"] = smooth_loss.item<float>();
    components["align"] = align_loss.item<float>();
    components["coverage"] = coverage_loss.item<float>();
    components["overlap"] = overlap_loss.item<float>();
    components["repetition"] = repetition_penalty.item<float>();
    components["stagnation"] = stagnation_penalty.item<float>();
    components["exploration"] = exploration_reward.item<float>();
    components["dynamic_smooth_weight"] = dynamic_smooth_weight;
    components["dynamic_align_weight"] = dynamic_align_weight;

    return { total_loss, components };
}




};



// ѵ������
std::shared_ptr<PathPlannerModel> train_model(int num_epochs = 100, int batch_size = 16,
    int width = 32, int height = 32) {
    int subpixel_scale = 3;//�������ܶ�
    // �������ݼ������ݼ�����
    auto dataset = FieldDataset(192 , width, height)
        .map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );


    // ��ʼ��ģ�͡���ʧ�������Ż���
    auto model = std::make_shared<PathPlannerModel>(width, height);
    PathLoss criterion(0.5f, 1.0f, 2.0f, 0.5f);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-4)
        .weight_decay(1e-5)
        .betas({ 0.9, 0.999 })  // ��������������Ĭ��ֵ���ɱ��֣�
    );

    //torch::optim::StepLR scheduler(optimizer, 5, 0.8);

    //torch::optim::CosineAnnealingLR scheduler(optimizer, num_epochs, 1e-5);

    torch::optim::ReduceLROnPlateauScheduler scheduler(
        optimizer,
        torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min,  // �����ָ��ֹͣ�½�ʱ����
        0.5f,                                            // ѧϰ�ʵ������� (new_lr = lr * factor)
        5,                                               // ����5��epoch�޸���
        1e-4,                                            // �������Ƶ���ֵ
        torch::optim::ReduceLROnPlateauScheduler::ThresholdMode::rel,  // �����ֵģʽ
        0,                                               // ��ȴʱ�䣨������ȴ����ٸ�epoch�ټ�أ�
        std::vector<float>(),                            // ÿ�����������Сѧϰ��
        1e-8,                                            // ѧϰ����С�仯
        true                                             // ��ӡѧϰ�ʵ�����Ϣ
    );

    // �豸����
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        model->to(device);
        std::cout << "Using CUDA for training" << std::endl;
    }
    else {
        std::cout << "Using CPU for training" << std::endl;
    }
    float best_loss = std::numeric_limits<float>::max();
    std::shared_ptr<PathPlannerModel> best_model = nullptr;

    // ·��������Ϊ����������*��
    int path_length = static_cast<int>(width * height * subpixel_scale * subpixel_scale * 0.5);

    // ѵ��ѭ��
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch:" << epoch + 1 << std::endl;
        model->train();
        float total_loss = 0.0f;
        //������ʧ�ؼ�Լ��
        std::unordered_map<std::string, float> loss_components = {
            {"smooth", 0}, {"align", 0}, {"coverage", 0},
            {"overlap", 0}, {"repetition", 0}, {"stagnation", 0}, {"exploration", 0}
        };
        int batch_count = 0;

        for (const auto& batch : *dataloader) {
            auto field = batch.data.to(device);
            auto start_pos = batch.target.to(device);
            int current_batch_size = field.size(0);

            // ��ʼ��·�����ѷ��ʼ���
            std::vector<torch::Tensor> path;
            path.push_back(start_pos);

            auto visited_count = torch::zeros({ current_batch_size, width * height }, device = device);
            std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt;

            // ����·��
            for (int i = 0; i < path_length - 1; ++i) {
                if (i % 100 == 0) {
                    std::cout << "Step " << i << "/" << path_length << std::endl;
                }
                auto current_pos = path.back();
                if (torch::isnan(current_pos).any().item<bool>() ||
                    torch::isinf(current_pos).any().item<bool>()) {
                    std::cout << "Invalid position detected at step " << i << std::endl;
                    break;
                }

                // �����ѷ��ʼ���
                auto pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
                auto pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
                auto pixel_ids = pixel_x * height + pixel_y;

                auto current_visited = visited_count.clone();
                current_visited.scatter_add_(1, pixel_ids.unsqueeze(1),
                    torch::ones({ current_batch_size, 1 }, device = device));

                // Ԥ����һ��λ��

                auto [next_pos, new_hidden] = model->forward(
                    field, current_pos, current_visited, hidden_state
                );
                // ========== ���������Լ�� ==========
               

                // ����������ӳ�䵽��������������ĵ�
                auto next_pos_subpixel = next_pos * subpixel_scale;

                // �ҵ�ÿ���������Ļ�����
                auto base_pixel_x = (next_pos_subpixel.index({ torch::indexing::Slice(), 0 }) / subpixel_scale).floor() * subpixel_scale;
                auto base_pixel_y = (next_pos_subpixel.index({ torch::indexing::Slice(), 1 }) / subpixel_scale).floor() * subpixel_scale;

                // 9����ѡ������ƫ��
                torch::Tensor candidate_offsets = torch::tensor({
                    {1, 1}, {1, 0}, {1, 2},
                    {0, 1}, {0, 0}, {0, 2},
                    {2, 1}, {2, 0}, {2, 2}
                    }, torch::dtype(torch::kFloat).device(device));

                // ��չά���Խ�����������
                base_pixel_x = base_pixel_x.unsqueeze(1).expand({ current_batch_size, 9 });
                base_pixel_y = base_pixel_y.unsqueeze(1).expand({ current_batch_size, 9 });
                candidate_offsets = candidate_offsets.unsqueeze(0).expand({ current_batch_size, 9, 2 });

                // �������к�ѡ������
                auto candidate_x = base_pixel_x + candidate_offsets.index({ torch::indexing::Slice(), torch::indexing::Slice(), 0 });
                auto candidate_y = base_pixel_y + candidate_offsets.index({ torch::indexing::Slice(), torch::indexing::Slice(), 1 });

                // ���㵽ÿ����ѡ��ľ���
                auto current_pos_expanded = current_pos.unsqueeze(1).expand({ current_batch_size, 9, 2 });
                auto next_pos_expanded = next_pos_subpixel.unsqueeze(1).expand({ current_batch_size, 9, 2 });

                auto candidate_points = torch::stack({ candidate_x, candidate_y }, 2);
                auto distances = (next_pos_expanded - candidate_points).norm(2, 2);

                // ѡ������ĺ�ѡ��
                auto min_indices = distances.argmin(1);
                auto selected_candidates = candidate_points.index({ torch::arange(current_batch_size), min_indices });

                // ת����ԭʼ����߶�
                next_pos = selected_candidates / subpixel_scale;
                // ========== ������Լ������ ==========
               


                path.push_back(next_pos);
                visited_count = current_visited;
                hidden_state = new_hidden;
                /*if (i > 0 && i % 200 == 0) {
                    hidden_state = std::nullopt;
                    std::cout << "Reset hidden state at step " << i << std::endl;
                }*/
            }

            // ��·��ת��Ϊ����
            auto path_tensor = torch::stack(path, 1);

            // ������ʧ
            auto [loss, components] = criterion(path_tensor, field, width, height);

            // ���򴫲����Ż�
            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
            optimizer.step();

            // �ۼ���ʧ
            total_loss += loss.item<float>();
            for (const auto& [key, value] : components) {
                if (key != "total") {
                    loss_components[key] += value;
                }
            }
            batch_count++;
        }


        // ����ƽ����ʧ
        float avg_loss = total_loss / batch_count;
        for (auto& [key, value] : loss_components) {
            value /= batch_count;
        }

        // ����ѧϰ��
        scheduler.step(avg_loss);

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            best_model = std::make_shared<PathPlannerModel>(*model); // ���
            std::cout << "New best model saved with loss: " << best_loss << std::endl;
        }

        // ��ӡѵ������
        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "]" << std::endl;
            std::cout << "  Total Loss: " << avg_loss << std::endl;
            std::cout << "  Components:" << std::endl;
            std::cout << "    Smooth: " << loss_components["smooth"]
                << ", Align: " << loss_components["align"] << std::endl;
            std::cout << "    Coverage: " << loss_components["coverage"]
                << ", Overlap: " << loss_components["overlap"] << std::endl;
            std::cout << "  Learning Rate: " << optimizer.param_groups()[0].options().get_lr() << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }
    //������ѵ�ģ��
    if (best_model) {
        std::cout << "Returning best model with loss: " << best_loss << std::endl;
        return best_model;
    }
    else {
        std::cout << "No valid model found, returning last model." << std::endl;
        return model;
    }
}





// ���ӻ����
//void visualize_results(const std::shared_ptr<PathPlannerModel>& model, int width = 32, int height = 32) {
//    // ������������
//    FieldDataset dataset(1, width, height);
//    auto example = dataset.get(0);
//    auto field = example.data.unsqueeze(0); // �������ά��
//    auto start_pos = example.target.unsqueeze(0);
//
//    // �豸����
//    torch::Device device(torch::kCPU);
//    if (torch::cuda::is_available()) {
//        device = torch::Device(torch::kCUDA);
//        field = field.to(device);
//        start_pos = start_pos.to(device);
//    }
//
//    // �л�������ģʽ
//    model->eval();
//    int path_length = static_cast<int>(width * height * 0.5);
//
//    // ����·��
//    std::vector<torch::Tensor> path;
//    path.push_back(start_pos);
//
//    auto visited_count = torch::zeros({ 1, width * height }, device = device);
//    std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt;
//
//    torch::NoGradGuard no_grad; // �����ݶȼ���
//    for (int i = 0; i < path_length - 1; ++i) {
//        auto current_pos = path.back();
//
//        // �����ѷ��ʼ���
//        auto pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
//        auto pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
//        auto pixel_ids = pixel_x * height + pixel_y;
//
//        auto current_visited = visited_count.clone();
//        current_visited.scatter_add_(1, pixel_ids.unsqueeze(1),
//            torch::ones({ 1, 1 }, device = device));
//
//        // Ԥ����һ��λ��
//        auto [next_pos, new_hidden] = model->forward(
//            field, current_pos, current_visited, hidden_state
//        );
//
//        path.push_back(next_pos);
//        visited_count = current_visited;
//        hidden_state = new_hidden;
//    }
//
//    // ת��ΪCPU�����ӻ�
//    auto path_tensor = torch::stack(path, 1).cpu();
//    auto field_cpu = field.squeeze(0).cpu();
//    auto start_pos_cpu = start_pos.squeeze(0).cpu();
//
//    // ��ȡ·����
//    std::vector<double> path_x, path_y;
//    for (int i = 0; i < path_tensor.size(1); ++i) {
//        path_x.push_back(path_tensor[0][i][0].item<float>());
//        path_y.push_back(path_tensor[0][i][1].item<float>());
//    }
//
//    // ���Ƴ�������ϡ����ʾ��
//    //namespace matplotlibcpp;
//    auto fig = plt::figure(true);
//    plt::axis("equal");
//
//    int step = std::max(1, width / 15);
//    std::vector<double> x, y, u, v;
//
//    for (int i = 0; i < height; i += step) {
//        for (int j = 0; j < width; j += step) {
//            x.push_back(j);
//            y.push_back(height - 1 - i);
//            u.push_back(field_cpu[0][i][j].item<float>()*0.5);
//            v.push_back(field_cpu[1][i][j].item<float>()*0.5);
//        }
//
//        //// ����������ӳ��
//        //std::map<std::string, std::string> quiver_props;
//        //quiver_props["color"] = "lightblue";
//        //quiver_props["linewidth"] = "0.5";
//
//        //// ��������ӳ��
//        //plt::quiver(x, y, u, v, quiver_props);
//    }
//
//    std::map<std::string, std::string> quiver_props;
//    quiver_props["color"] = "lightblue";
//    quiver_props["linewidth"] = "0.5";
//
//    // ��������ӳ��
//    plt::quiver(x, y, u, v, quiver_props);
//    //plt::quiver(x, y, u, v, { {"color", "lightblue"}, {"linewidth", 0.5} });
//
//    // ����·��
//
//    //plt::plot(path_x, path_y, "r-", { {"linewidth", 2} });
//    std::string p_for = "r-";
//    plt::plot(path_x, path_y, p_for);
//   
//
//    // ��������յ�
//    std::vector<double> start_x = { path_x[0] };
//    std::vector<double> start_y = { path_y[0] };
//    std::map<std::string, std::string> start_props;
//    start_props["marker"] = "*";
//    start_props["color"] = "green";
//
//    std::vector<double> end_x = { path_x.back() };
//    std::vector<double> end_y = { path_y.back() };
//    std::map<std::string, std::string> end_props;
//    end_props["marker"] = "x";
//    end_props["color"] = "red";
//
//
//    plt::scatter(start_x, start_y, 50.0, "green","*");
//    plt::scatter(end_x, end_y, 50.0,end_props);
//
//    plt::xlabel("X");
//    plt::ylabel("Y");
//    plt::title("Path Generated by Self-Supervised Model");
//    plt::xlim( 0, width - 1);
//    plt::ylim( 0, height - 1);
//    plt::grid(true);
//    plt::show();
//}

// �޸ĵ�visualize_results�����������￪ʼ�滻��

void visualize_results(const std::shared_ptr<PathPlannerModel>& model, int width = 32, int height = 32) {
    try {
        // ������������ - �ⲿ�ֱ��ֲ���
        FieldDataset dataset(1, width, height);
        auto example = dataset.get(0);
        auto field = example.data.unsqueeze(0); // �������ά��
        auto start_pos = example.target.unsqueeze(0);

        // �豸���� - ���ֲ���
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            field = field.to(device);
            start_pos = start_pos.to(device);
        }

        // �л�������ģʽ - ���ֲ���
        model->eval();
        int path_length = static_cast<int>(width * height * 8);

        // ����·�� - ���ֲ���
        std::vector<torch::Tensor> path;
        path.push_back(start_pos);

        auto visited_count = torch::zeros({ 1, width * height }, device = device);
        std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt;

        torch::NoGradGuard no_grad;
        for (int i = 0; i < path_length - 1; ++i) {
            auto current_pos = path.back();

            // �����ѷ��ʼ���
            auto pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
            auto pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
            auto pixel_ids = pixel_x * height + pixel_y;

            auto current_visited = visited_count.clone();
            current_visited.scatter_add_(1, pixel_ids.unsqueeze(1),
                torch::ones({ 1, 1 }, device = device));

            // Ԥ����һ��λ��
            auto [next_pos, new_hidden] = model->forward(
                field, current_pos, current_visited, hidden_state
            );

            path.push_back(next_pos);
            visited_count = current_visited;
            hidden_state = new_hidden;
        }

        // ========== �����￪ʼ�滻ΪOpenCV�汾 ==========

        // ת��ΪCPU
        auto path_tensor = torch::stack(path, 1).cpu();
        auto field_cpu = field.squeeze(0).cpu();

        // ����OpenCVͼ�񣨷Ŵ���ʾ��
        int scale = 60; // �Ŵ���
        cv::Mat image = cv::Mat::ones(height * scale, width * scale, CV_8UC3) * 255;

        // ��������
        for (int i = 0; i <= height; ++i) {
            cv::line(image,
                cv::Point(0, i * scale),
                cv::Point(width * scale, i * scale),
                cv::Scalar(200, 200, 200), 1);
        }
        for (int j = 0; j <= width; ++j) {
            cv::line(image,
                cv::Point(j * scale, 0),
                cv::Point(j * scale, height * scale),
                cv::Scalar(200, 200, 200), 1);
        }

        // ����������
        int step = std::max(1, width / 10);
        for (int i = 0; i < height; i += step) {
            for (int j = 0; j < width; j += step) {
                float u = field_cpu[0][i][j].item<float>();
                float v = field_cpu[1][i][j].item<float>();

                // �����ͷ�����յ�
                cv::Point start(j * scale + scale / 2, i * scale + scale / 2);
                cv::Point end(start.x + u * scale * 0.4f, start.y + v * scale * 0.4f);

                // ���Ƽ�ͷ
                cv::arrowedLine(image, start, end, cv::Scalar(100, 100, 255), 2, 8, 0, 0.3);
            }
        }

        // ��ȡ·����
        std::vector<cv::Point> path_points;
        for (int i = 0; i < path_tensor.size(1); ++i) {
            float x = path_tensor[0][i][0].item<float>();
            float y = path_tensor[0][i][1].item<float>();
            path_points.push_back(cv::Point(x * scale, y * scale));
        }

        // ����·����
        for (size_t i = 1; i < path_points.size(); ++i) {
            cv::line(image, path_points[i - 1], path_points[i], cv::Scalar(255, 0, 0), 2);
        }
        //==========================��Ҫ�޸�Ϊ����·�������ǵ�============================
        // ����·����
       /* for (const auto& point : path_points) {
            cv::circle(image, point, 3, cv::Scalar(0, 0, 255), -1);
        }*/
        cv::Mat path_image = image.clone(); // ����·���ĵ���ͼ��

        // ʹ�ø��ֵ�����������Ҫ·��
        for (size_t i = 1; i < path_points.size(); ++i) {
            // ����·���εĳ��ȵ���������ϸ���ϳ��Ķ��ýϴֵ��ߣ�
            double segment_length = cv::norm(path_points[i] - path_points[i - 1]);
            int thickness = std::max(2, static_cast<int>(segment_length / 10));

            // ʹ�ý���ɫ������ɫ����㣩����ɫ���м䣩����ɫ���յ㣩
            double progress = static_cast<double>(i) / path_points.size();
            cv::Scalar color;
            if (progress < 0.33) {
                // ��ɫ����ɫ�Ľ���
                color = cv::Scalar(255 * (0.33 - progress) * 3, 255, 0);
            }
            else if (progress < 0.66) {
                // ��ɫ����ɫ�Ľ���
                color = cv::Scalar(0, 255 * (0.66 - progress) * 3, 255);
            }
            else {
                // ��ɫ����ɫ�Ľ���
                color = cv::Scalar(255 * (progress - 0.66) * 3, 0, 255);
            }

            cv::line(path_image, path_points[i - 1], path_points[i], color, thickness);
            cv::circle(path_image, path_points[i], 2, cv::Scalar(0, 0, 255), -1); // ��ɫСԲ��
            // ÿ��һ��������Ʒ����ͷ
            static double accumulated_distance = 0;
            accumulated_distance += segment_length;

            // ���ü�ͷ���������ÿ50���ػ���һ����ͷ��
            const double arrow_interval = 50.0;

            if (accumulated_distance >= arrow_interval) {
                // �����ͷλ�ã����߶��е����ݱ�����
                double arrow_ratio = 0.5; // ��ͷ���߶��ϵ�λ�ñ�����0-1֮�䣩
                cv::Point arrow_pos = path_points[i - 1] +
                    cv::Point((path_points[i] - path_points[i - 1]) * arrow_ratio);

                // �����߶η�������
                cv::Point direction = path_points[i] - path_points[i - 1];
                double angle = atan2(direction.y, direction.x) * 180 / CV_PI;

                // ������ɫС��ͷ
                cv::Scalar arrow_color(255, 0, 255); // ��ɫ(BGR��ʽ)
                int arrow_length = 15; // ��ͷ����
                int arrow_thickness = 5; // ��ͷ��ϸ

                // ʹ��arrowedLine�������Ƽ�ͷ
                cv::arrowedLine(path_image,
                    arrow_pos - cv::Point(arrow_length / 2 * cos(angle * CV_PI / 180),
                        arrow_length / 2 * sin(angle * CV_PI / 180)),
                    arrow_pos + cv::Point(arrow_length / 2 * cos(angle * CV_PI / 180),
                        arrow_length / 2 * sin(angle * CV_PI / 180)),
                    arrow_color, arrow_thickness, 8, 0, 0.3);

                // �����ۻ�����
                accumulated_distance = 0;

            }

        }

        for (size_t i = 0; i < path_points.size(); ++i) {
            // ʹ�ð�͸���ĺ�ɫԲ�㣬�����غϵĵ����ʾ�ø���
            cv::Scalar point_color(0, 0, 255); // ��ɫ
            int point_radius = 10;

            // ������ͼ���ϻ��Ƶ㣨�ڻ��֮��
            cv::circle(image, path_points[i], point_radius, point_color, -1);

            // �������ܼ�������ÿ�����������һ��������̫��
            // if (i % 5 == 0) { // ÿ5���㻭һ��
            //     cv::circle(image, path_points[i], point_radius, point_color, -1);
            // }
        }

        double alpha = 0.7; // ·��͸����
        cv::addWeighted(path_image, alpha, image, 1 - alpha, 0, image);
        //���Ƽ�ͷ��ʾ����




        // ��������յ�
        if (!path_points.empty()) {
            // ��㣨��ɫ��
            cv::circle(image, path_points.front(), 8, cv::Scalar(0, 255, 0), -1);
            cv::putText(image, "Start", path_points.front() + cv::Point(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // �յ㣨��ɫ��
            cv::circle(image, path_points.back(), 8, cv::Scalar(0, 0, 255), -1);
            cv::putText(image, "End", path_points.back() + cv::Point(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        // ��ӱ������Ϣ
        cv::putText(image, "Path Planning Visualization", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        cv::putText(image, "Blue: Path, Pink: Vector Field", cv::Point(10, height * scale - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        for (cv::Point i: path_points) {
            std::cout << i << std::endl;
        }
    
        // ��ʾͼ��
        cv::imshow("Path Planning Result", image);
        std::cout << "Press any key to close the window..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();

        // ����ͼ��
        cv::imwrite("path_planning_result.png", image);
        std::cout << "Image saved as path_planning_result.png" << std::endl;

    }
    

    catch (const std::exception& e) {
        std::cout << "Error in visualization: " << e.what() << std::endl;
    }
}

int main() {
    // �����������
    torch::manual_seed(42);

    // ѵ��ģ��
    int width = 8, height = 8;
    std::cout << "Training model for grid size " << width << "x" << height << "..." << std::endl;
    auto model = train_model(50, 16, width, height);

    // ���ӻ����
    std::cout << "Visualizing results..." << std::endl;
    visualize_results(model, width, height);

    return 0;
}
