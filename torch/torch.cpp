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
// 数据集类：生成不同类型的曲线场
class FieldDataset : public torch::data::Dataset<FieldDataset> {
private:
    int num_samples_;
    int width_;
    int height_;
    std::mt19937 rng_;

public:
    // 构造函数
    FieldDataset(int num_samples, int width, int height)
        : num_samples_(num_samples), width_(width), height_(height) {
        std::random_device rd;
        rng_.seed(rd());
    }

    // 返回样本数
    torch::data::Example<> get(size_t index) override {
        // 随机选择场类型
        std::uniform_int_distribution<int> dist_type(0, 2);
        int field_type = dist_type(rng_);

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
        else if (field_type == 1) { // 旋转场
            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float dx = x[j];
                    float dy = y[i];
                    float radius = std::sqrt(dx * dx + dy * dy);
                    if (radius < 0.1f) radius = 0.1f;

                    float angle = std::atan2(dy, dx) + M_PI / 2.0f;

                    // 随机选择旋转方向
                    std::uniform_real_distribution<float> dist_dir(0, 1);
                    if (dist_dir(rng_) > 0.5f) {
                        angle = -angle;
                    }

                    field_x[i][j] = std::cos(angle) / radius;
                    field_y[i][j] = std::sin(angle) / radius;
                }
            }

            // 归一化
            auto magnitude = torch::sqrt(field_x.square() + field_y.square());
            field_x = field_x / magnitude;
            field_y = field_y / magnitude;
        }
        else { // 梯度场
            std::uniform_real_distribution<float> dist_angle(0, 2 * M_PI);
            float angle_start = dist_angle(rng_);
            float angle_end = fmod(angle_start + M_PI / 2.0f, 2 * M_PI);

            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    float weight = (x[j] + 1.0f) / 2.0f; // 从左到右从0过渡到1
                    field_x[i][j] = std::cos(angle_start) * (1 - weight) + std::cos(angle_end) * weight;
                    field_y[i][j] = std::sin(angle_start) * (1 - weight) + std::sin(angle_end) * weight;
                }
            }
        }

        // 组合场向量并调整形状为[2, height, width]
        auto field = torch::stack({ field_x, field_y }).to(torch::kFloat32);

        // 随机选择起点（中心区域）
        /*std::uniform_real_distribution<float> dist_x(width_ * 0.3f, width_ * 0.7f);
        std::uniform_real_distribution<float> dist_y(height_ * 0.3f, height_ * 0.7f);
        auto start_pos = torch::tensor({ dist_x(rng_), dist_y(rng_) }, torch::kFloat32);*/
        //固定1，1
        auto start_pos = torch::tensor({ 0.5f, 0.5f }, torch::kFloat32);
        return { field, start_pos };
    }

    // 返回数据集大小
    torch::optional<size_t> size() const override {
        return num_samples_;
    }
};

// 路径规划模型：CNN + LSTM
//TORCH_MODULE (PathPlannerModel);
class PathPlannerModel : public torch::nn::Module {
private:
    int width_;
    int height_;
    int total_pixels_;
    int hidden_dim_;

    // CNN层：提取场特征
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

    // LSTM层：生成路径序列
    torch::nn::LSTM lstm{
        torch::nn::LSTMOptions(128 + 2 + 32, 128)
            .num_layers(2)
            .batch_first(true)
            .dropout(0.2)
    };

    // 输出层：预测下一个位置
    torch::nn::Sequential output_head{
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(64, 2)
    };

    // 已访问像素的嵌入层
    torch::nn::Embedding visited_embedding{
        torch::nn::EmbeddingOptions(1024 + 1, 32) // 最大像素数设为1024
    };

public:
    // 构造函数
    PathPlannerModel(int width, int height, int hidden_dim = 128)
        : width_(width), height_(height), hidden_dim_(hidden_dim) {
        total_pixels_ = width * height;

        // 注册模块
        register_module("cnn", cnn);
        register_module("lstm", lstm);
        register_module("output_head", output_head);
        register_module("visited_embedding", visited_embedding);

        // 初始化权重
        init_weights();
    }

    // 初始化权重
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

    // 前向传播
    std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
        forward(
            const torch::Tensor& field_,
            const torch::Tensor& current_pos,
            const torch::Tensor& visited_count,
            std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt
        ) {
        int batch_size = field_.size(0);
        int subpixel_scale = 3;
        //异常检测
        if (torch::isnan(field_).any().item<bool>()) {
            throw std::runtime_error("NaN detected in field input");
        }
        if (torch::isnan(current_pos).any().item<bool>()) {
            throw std::runtime_error("NaN detected in current_pos input");
        }
        auto field = field_.contiguous();
        // 1. 使用CNN提取场特征
        //std::cout << field.sizes() << std::endl;

        auto field_features = cnn->forward(field); // [batch, hidden_dim, height, width]

        // 2. 获取当前位置的场特征（双线性插值）
        //std::cout << current_pos << std::endl;

        auto x_norm = (current_pos.index({ torch::indexing::Slice(), 0 }) / (width_ - 1)) * 2 - 1;
        auto y_norm = (current_pos.index({ torch::indexing::Slice(), 1 }) / (height_ - 1)) * 2 - 1;

        auto grid = torch::stack({ x_norm, y_norm }, -1).unsqueeze(1).unsqueeze(1); // [batch, 1,1, 2]

        auto current_features = torch::nn::functional::grid_sample(
            field_features,
            grid,
            torch::nn::functional::GridSampleFuncOptions()

            .mode(torch::kBilinear)//双线性插值得到平滑特征
            .padding_mode(torch::kZeros)
            .align_corners(true)//坐标对齐到像素角落
        );

        current_features = current_features.squeeze(-1).squeeze(-1); // [batch, hidden_dim]

        //std::cout << current_features.sizes() << std::endl;
        // 3. 处理已访问信息
        auto visited_num = (visited_count > 0).to(torch::kFloat32).sum(1).to(torch::kLong);
        auto visited_feat = visited_embedding->forward(visited_num); // [batch, 32]

        // 4. 准备LSTM输入
        auto normalized_pos = current_pos / torch::tensor({ static_cast<float>(width_), static_cast<float>(height_) },
            torch::dtype(torch::kFloat32).device(current_pos.device()));
        auto lstm_input = torch::cat({ current_features, normalized_pos, visited_feat }, 1).unsqueeze(1);

        // 5. LSTM前向传播
        auto lstm_out = lstm->forward(lstm_input, hidden_state);

        // 6. 预测下一个位置的偏移
        auto delta = output_head->forward(std::get<0>(lstm_out).squeeze(1));

        // 限制最大偏移
       delta = torch::tanh(delta) * (2.0f/subpixel_scale);

       // ========== 修改最小偏移限制 ==========
   // 基础最小偏移（基于子像素网格调整）
       float base_min_offset = 0.1f; // 进一步减小，因为现在有更多可选点

       // 计算移动方向与场方向的一致性
       auto grid_x_norm = (current_pos.index({ torch::indexing::Slice(), 0 }) / (width_ - 1)) * 2 - 1;
       auto grid_y_norm = (current_pos.index({ torch::indexing::Slice(), 1 }) / (height_ - 1)) * 2 - 1;
       auto pos_grid = torch::stack({ grid_x_norm, grid_y_norm }, -1).unsqueeze(1).unsqueeze(1);

       // 获取当前位置的场方向
       auto current_field = torch::nn::functional::grid_sample(
           field, pos_grid,
           torch::nn::functional::GridSampleFuncOptions()
           .mode(torch::kBilinear)
           .padding_mode(torch::kZeros)
           .align_corners(true)
       ).squeeze(-1).squeeze(-1);

       // 归一化场方向
       auto field_norm = torch::norm(current_field, 2, -1, true) + 1e-8f;
       auto field_dir = current_field / field_norm;

       // 归一化移动方向
       auto delta_norm = torch::norm(delta, 2, -1, true) + 1e-8f;
       auto delta_dir = delta / delta_norm;

       // 计算方向一致性（点积）
       auto alignment = (field_dir * delta_dir).sum(-1);

       // 动态调整最小偏移
       auto min_offset = torch::full({ batch_size }, base_min_offset,
           torch::dtype(torch::kFloat32).device(delta.device()));

       // 获取当前位置在子像素网格中的索引
       auto current_pos_subpixel = current_pos * subpixel_scale;
       auto current_subpixel_x = (current_pos_subpixel.index({ torch::indexing::Slice(), 0 })).clamp(0, width_ * subpixel_scale - 1).to(torch::kLong);
       auto current_subpixel_y = (current_pos_subpixel.index({ torch::indexing::Slice(), 1 })).clamp(0, height_ * subpixel_scale - 1).to(torch::kLong);
       auto current_subpixel_ids = current_subpixel_x * (height_ * subpixel_scale) + current_subpixel_y;

       // 更智能的偏移调整策略（基于子像素访问）
       for (int b = 0; b < batch_size; ++b) {
           // 计算当前位置在原始像素中的索引
           auto pixel_x = current_pos.index({ b, 0 }).clamp(0, width_ - 1).to(torch::kLong).item<int>();
           auto pixel_y = current_pos.index({ b, 1 }).clamp(0, height_ - 1).to(torch::kLong).item<int>();
           auto pixel_id = pixel_x * height_ + pixel_y;

           int visit_count = visited_count[b][pixel_id].item<int>();
           float align_score = alignment[b].item<float>();

           // 调整策略保持不变，但阈值可能需要微调
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

       // 应用最小偏移限制
       auto delta_magnitude = torch::norm(delta, 2, -1, true);
       auto enforced_magnitude = torch::max(delta_magnitude, min_offset.unsqueeze(-1));
       delta = delta_dir * enforced_magnitude;
       // ========== 最小偏移限制结束 ==========
      
        // 计算下一个位置
        auto next_pos = current_pos + delta;

        // 增强边界处理
        auto near_boundary_x = (next_pos.index({ torch::indexing::Slice(), 0 }) < 1.0f) |
            (next_pos.index({ torch::indexing::Slice(), 0 }) > width_ - 2.0f);
        auto near_boundary_y = (next_pos.index({ torch::indexing::Slice(), 1 }) < 1.0f) |
            (next_pos.index({ torch::indexing::Slice(), 1 }) > height_ - 2.0f);
        auto near_boundary = near_boundary_x | near_boundary_y;

        // 如果靠近边界，给予反弹力
        auto boundary_force = torch::zeros_like(delta);
        boundary_force.index({ near_boundary_x, 0 }) = (width_ / 2.0f - next_pos.index({ near_boundary_x, 0 })) * 0.1f;
        boundary_force.index({ near_boundary_y, 1 }) = (height_ / 2.0f - next_pos.index({ near_boundary_y, 1 })) * 0.1f;

        next_pos = next_pos + boundary_force;

        // 确保位置在网格范围内
        float margin = 0.5f / subpixel_scale; // 边界留出半个子像素的余地
        next_pos = torch::clamp(
            next_pos,
            torch::tensor(margin, torch::device(next_pos.device())),
            torch::tensor({ static_cast<float>(width_ - 1 - margin),
                          static_cast<float>(height_ - 1 - margin) },
                torch::device(next_pos.device()))
        );
        // ========== 添加探索噪声（仅在训练时） ==========
        if (this->is_training()) {
            // 计算当前位置的像素索引
            auto current_pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width_ - 1).to(torch::kLong);
            auto current_pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height_ - 1).to(torch::kLong);
            auto current_pixel_ids = current_pixel_x * height_ + current_pixel_y;

            // 基于访问次数的自适应噪声
            auto visit_counts = visited_count.gather(1, current_pixel_ids.unsqueeze(1)).squeeze(1);
            auto noise_scale = torch::where(
                visit_counts > 10,
                torch::tensor(0.5f, torch::device(next_pos.device())),  // 高访问区域增加噪声
                torch::tensor(0.1f, torch::device(next_pos.device()))   // 低访问区域减少噪声
            );

            auto exploration_noise = torch::randn_like(next_pos) * noise_scale.unsqueeze(1);
            next_pos = next_pos + exploration_noise;

            // 再次确保位置在网格范围内
            next_pos = torch::clamp(
                next_pos,
                torch::tensor(0.5f, torch::device(next_pos.device())),
                torch::tensor({ static_cast<float>(width_ - 1.5), static_cast<float>(height_ - 1.5) },
                    torch::device(next_pos.device()))
            );
        }
        // ========== 探索噪声结束 ==========

        return { next_pos, std::get<1>(lstm_out) };
    }
};

// 自定义损失函数
class PathLoss {
private:
    float smooth_weight_;
    float align_weight_;
    float coverage_weight_;
    float overlap_weight_;

public:
    // 构造函数
    PathLoss(float smooth_weight = 1.0f, float align_weight = 1.0f,
        float coverage_weight = 10.0f, float overlap_weight = 0.5f)
        : smooth_weight_(smooth_weight), align_weight_(align_weight),
        coverage_weight_(coverage_weight), overlap_weight_(overlap_weight) {
    }

    // 计算损失
    //std::pair<torch::Tensor, std::unordered_map<std::string, float>>
    //    operator()(const torch::Tensor& path, const torch::Tensor& field, int width, int height) {
    //    int batch_size = path.size(0);
    //    int seq_len = path.size(1);
    //    int total_pixels = width * height;

    //    // 1. 平滑性损失
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

    //    // 2. 场对齐损失
    //    torch::Tensor align_loss = torch::tensor(0.0f, device = path.device());
    //    if (seq_len > 1) {
    //        //算路径相邻点的中点
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
    //        //上面是从从路径中采样（归一）
    //        // auto sampled_field_normalized = torch::nn::functional::normalize(sampled_field, 2, 2);

    //        auto sampled_field_normalized = torch::nn::functional::normalize(
    //            sampled_field,
    //            torch::nn::functional::NormalizeFuncOptions()
    //            .p(2)          // 范数的阶数（对应第一个整数参数2）
    //            .dim(2)        // 要规范化的维度（对应第二个整数参数2）
    //        );

    //        //计算路径移动的方向
    //        auto dirs = path.index({ torch::indexing::Slice(), torch::indexing::Slice(1, seq_len) }) -
    //            path.index({ torch::indexing::Slice(), torch::indexing::Slice(0, seq_len - 1) });
    //        auto dir_magnitudes = dirs.norm(2, 2, true) + 1e-8f;
    //        auto dirs_normalized = dirs / dir_magnitudes;

    //        auto dot_product = (dirs_normalized * sampled_field).sum(-1);
    //        //这里注释掉的原因是这个逆着场方向惩罚很多。
    //        /*auto alignment = (dirs_normalized * sampled_field_normalized).sum(2);
    //        align_loss = (1 - alignment).mean();*/
    //        //用sin均值
    //        auto cross_product = torch::abs(
    //            dirs_normalized.index({ "...", 0 }) * sampled_field_normalized.index({ "...", 1 }) -
    //            dirs_normalized.index({ "...", 1 }) * sampled_field_normalized.index({ "...", 0 })
    //        );//惩罚垂直移动
    //        auto align_loss_m = 1.0 - dot_product.abs();//惩罚角度偏差
    //        align_loss = (cross_product+ align_loss_m).mean();  // 直接用sinθ的均值作为损失
    //    }

    //    // 3. 覆盖度损失
    //    //auto pixel_x = path.index({ torch::indexing::Slice(), torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
    //    //auto pixel_y = path.index({ torch::indexing::Slice(), torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
    //    //auto pixel_ids = pixel_x * height + pixel_y;//计算唯一像素id
    //    // 
    //    ////统计每一个像素的访问次数
    //    //torch::Tensor visited = torch::zeros({ batch_size, total_pixels }, device = path.device());
    //    //for (int i = 0; i < seq_len; ++i) {
    //    //    auto ids = pixel_ids.index({ torch::indexing::Slice(), i }).unsqueeze(1);
    //    //    visited.scatter_add_(1, ids, torch::ones({ batch_size, 1 }, device = path.device()));
    //    //}

    //    //auto coverage = (visited > 0).to(torch::kFloat32).mean(1);//计算覆盖度
    //    //auto coverage_loss = (1 - coverage).mean();

    //    //// 4. 重叠惩罚损失
    //    //auto overlap = (visited - 1).clamp_min(0);
    //    //auto overlap_loss = overlap.mean();
    //     
    //    // 3. 覆盖度损失 - 改进版本（考虑路径宽度和线段覆盖）
    //    
    //   
    //    // 创建覆盖图，考虑路径宽度
    //    float path_width = 1.0f; // 路径宽度，可以根据需要调整
    //    torch::Tensor visited = torch::zeros({ batch_size, total_pixels }, device = path.device());

    //    for (int b = 0; b < batch_size; ++b) {
    //        for (int i = 0; i < seq_len - 1; ++i) {
    //            // 获取当前线段起点和终点
    //            auto p1 = path.index({ b, i });
    //            auto p2 = path.index({ b, i + 1 });

    //            float x1 = p1[0].item<float>();
    //            float y1 = p1[1].item<float>();
    //            float x2 = p2[0].item<float>();
    //            float y2 = p2[1].item<float>();

    //            // 计算线段方向向量
    //            float dx = x2 - x1;
    //            float dy = y2 - y1;
    //            float length = std::sqrt(dx * dx + dy * dy);

    //            // 自适应采样：根据线段长度决定采样点数
    //            int num_samples = std::max(2, static_cast<int>(length / 0.5f));

    //            for (int s = 0; s < num_samples; ++s) {
    //                float t = static_cast<float>(s) / (num_samples - 1);
    //                float x = x1 + t * dx;
    //                float y = y1 + t * dy;

    //                // 考虑路径宽度，标记周围像素
    //                int px = static_cast<int>(std::round(x));
    //                int py = static_cast<int>(std::round(y));

    //                // 标记路径宽度范围内的像素
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

    //    // 改进的覆盖度计算
    //    auto coverage_mask = (visited > 0).to(torch::kFloat32);
    //    auto coverage = coverage_mask.view({ batch_size, -1 }).mean(1);
    //    auto coverage_loss = (1 - coverage).mean();

    //    // 4. 改进的重叠惩罚损失 - 使用平滑惩罚函数
    //    auto overlap_count = visited - coverage_mask; // 减去基础覆盖（第一次访问）
    //    auto smooth_overlap = torch::log(1.0 + overlap_count.clamp_min(0)); // 对数平滑，避免过度惩罚
    //    auto overlap_loss = smooth_overlap.mean();

    //    // 总损失
    //    auto total_loss = smooth_weight_ * smooth_loss +
    //        align_weight_ * align_loss +
    //        coverage_weight_ * coverage_loss +
    //        overlap_weight_ * overlap_loss;

    //    // 损失分量（用于监控）
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

    // 子像素网格参数
    int subpixel_scale = 3; // 每个像素划分为3x3子像素
    int subpixel_width = width * subpixel_scale;
    int subpixel_height = height * subpixel_scale;
    int total_subpixels = subpixel_width * subpixel_height;

    torch::Device device = path.device();

    // 1. 平滑性损失（保持不变）
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

    // 2. 场对齐损失（保持不变）
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

    // 3. 新的覆盖度损失（基于子像素网格）
    torch::Tensor visited_subpixels = torch::zeros({ batch_size, total_subpixels }, device = device);
    torch::Tensor coverage_loss = torch::tensor(0.0f, device);
    torch::Tensor overlap_loss = torch::tensor(0.0f, device);
    torch::Tensor repetition_penalty = torch::tensor(0.0f, device);

    if (seq_len > 0) {
        // 将路径点映射到子像素网格
        auto path_subpixel = path * subpixel_scale; // 缩放路径点到子像素坐标

        // 记录每个点选择的子像素ID（用于重复惩罚）
        torch::Tensor selected_subpixel_ids = torch::zeros({ batch_size, seq_len },
            torch::dtype(torch::kLong).device(device));

        for (int b = 0; b < batch_size; ++b) {
            std::vector<int> visited_this_batch; // 记录本次batch中访问过的子像素

            for (int i = 0; i < seq_len; ++i) {
                // 获取当前路径点在子像素网格中的坐标
                float x = path_subpixel[b][i][0].item<float>();
                float y = path_subpixel[b][i][1].item<float>();

                // 找到最近的子像素中心点（9个候选点之一）
                int base_x = static_cast<int>(std::floor(x / subpixel_scale)) * subpixel_scale;
                int base_y = static_cast<int>(std::floor(y / subpixel_scale)) * subpixel_scale;

                // 9个候选点的相对偏移
                std::vector<std::pair<int, int>> candidate_offsets = {
                    {1, 1}, {1, 0}, {1, 2},  // 中间行
                    {0, 1}, {0, 0}, {0, 2},  // 上行
                    {2, 1}, {2, 0}, {2, 2}   // 下行
                };

                // 找到距离最近的候选点
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

                // 记录选择的子像素ID
                int subpixel_id = candidate_x * subpixel_height + candidate_y;
                selected_subpixel_ids[b][i] = subpixel_id;

                // 标记当前子像素及其8邻域为已覆盖
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

                // 检查是否重复选择（硬性规定惩罚）
                if (std::find(visited_this_batch.begin(), visited_this_batch.end(), subpixel_id) !=
                    visited_this_batch.end()) {
                    // 重复选择，给予重惩罚
                    repetition_penalty = repetition_penalty + 10.0f;
                }
                visited_this_batch.push_back(subpixel_id);
            }
        }

        // 计算覆盖度损失
        auto coverage_mask = (visited_subpixels > 0).to(torch::kFloat32);
        auto coverage = coverage_mask.view({ batch_size, -1 }).mean(1);
        coverage_loss = (1 - coverage).mean();

        // 计算重叠损失（只考虑8邻域的重叠，不包括中心点本身）
        auto center_visited = torch::zeros({ batch_size, total_subpixels }, device = device);

        // 只标记中心点（用于区分8邻域重叠）
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < seq_len; ++i) {
                int subpixel_id = selected_subpixel_ids[b][i].item<int>();
                center_visited[b][subpixel_id] = 1.0;
            }
        }

        // 重叠 = 总覆盖 - 中心点覆盖（即8邻域的重叠部分）
        auto overlap = visited_subpixels - center_visited;
        auto valid_overlap = torch::max(overlap - 1.0, torch::tensor(0.0f)); // 减去基础重叠
        overlap_loss = torch::log(1.0 + valid_overlap).mean();
    }

    // 4. 探索奖励（鼓励访问未探索区域）
    torch::Tensor exploration_reward = torch::tensor(0.0f, device);
    if (seq_len > 5) {
        // 基于覆盖度的探索奖励
        auto coverage_mask = (visited_subpixels > 0).to(torch::kFloat32);
        auto coverage = coverage_mask.view({ batch_size, -1 }).mean(1);
        exploration_reward = -coverage.mean() * 0.1f; // 覆盖度越低，奖励越高
    }

    // 5. 停滞检测和动态权重调整
    float dynamic_smooth_weight = smooth_weight_;
    float dynamic_align_weight = align_weight_;
    torch::Tensor stagnation_penalty = torch::tensor(0.0f, device);

    if (seq_len > 10) {
        // 检测最近几步是否停滞（在同一个像素内）
        int check_steps = std::min(5, seq_len / 3);
        auto recent_path = path.index({ torch::indexing::Slice(),
                                      torch::indexing::Slice(seq_len - check_steps, seq_len) });

        // 将路径点映射到像素坐标
        auto recent_pixels = recent_path / subpixel_scale;
        auto pixel_changes = (recent_pixels.index({ torch::indexing::Slice(),
                                                 torch::indexing::Slice(1, check_steps) }) -
            recent_pixels.index({ torch::indexing::Slice(),
                                 torch::indexing::Slice(0, check_steps - 1) })).norm(2, 2);

        // 如果最近几步平均移动小于0.3个像素，认为停滞
        if (pixel_changes.mean().item<float>() < 0.3f) {
            dynamic_smooth_weight = smooth_weight_ * 0.2f;
            dynamic_align_weight = align_weight_ * 0.2f;
            stagnation_penalty = torch::tensor(5.0f, device);
        }
    }

    // 总损失
    auto total_loss = dynamic_smooth_weight * smooth_loss +
        dynamic_align_weight * align_loss +
        coverage_weight_ * coverage_loss +
        overlap_weight_ * overlap_loss +
        repetition_penalty +
        stagnation_penalty +
        exploration_reward;

    // 损失分量（用于监控）
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



// 训练函数
std::shared_ptr<PathPlannerModel> train_model(int num_epochs = 100, int batch_size = 16,
    int width = 32, int height = 32) {
    int subpixel_scale = 3;//子像素密度
    // 创建数据集和数据加载器
    auto dataset = FieldDataset(192 , width, height)
        .map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );


    // 初始化模型、损失函数和优化器
    auto model = std::make_shared<PathPlannerModel>(width, height);
    PathLoss criterion(0.5f, 1.0f, 2.0f, 0.5f);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-4)
        .weight_decay(1e-5)
        .betas({ 0.9, 0.999 })  // 调整动量参数（默认值，可保持）
    );

    //torch::optim::StepLR scheduler(optimizer, 5, 0.8);

    //torch::optim::CosineAnnealingLR scheduler(optimizer, num_epochs, 1e-5);

    torch::optim::ReduceLROnPlateauScheduler scheduler(
        optimizer,
        torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min,  // 当监控指标停止下降时调整
        0.5f,                                            // 学习率调整因子 (new_lr = lr * factor)
        5,                                               // 容忍5个epoch无改善
        1e-4,                                            // 衡量改善的阈值
        torch::optim::ReduceLROnPlateauScheduler::ThresholdMode::rel,  // 相对阈值模式
        0,                                               // 冷却时间（调整后等待多少个epoch再监控）
        std::vector<float>(),                            // 每个参数组的最小学习率
        1e-8,                                            // 学习率最小变化
        true                                             // 打印学习率调整信息
    );

    // 设备配置
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

    // 路径长度设为像素总数的*倍
    int path_length = static_cast<int>(width * height * subpixel_scale * subpixel_scale * 0.5);

    // 训练循环
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch:" << epoch + 1 << std::endl;
        model->train();
        float total_loss = 0.0f;
        //更新损失控件约束
        std::unordered_map<std::string, float> loss_components = {
            {"smooth", 0}, {"align", 0}, {"coverage", 0},
            {"overlap", 0}, {"repetition", 0}, {"stagnation", 0}, {"exploration", 0}
        };
        int batch_count = 0;

        for (const auto& batch : *dataloader) {
            auto field = batch.data.to(device);
            auto start_pos = batch.target.to(device);
            int current_batch_size = field.size(0);

            // 初始化路径和已访问计数
            std::vector<torch::Tensor> path;
            path.push_back(start_pos);

            auto visited_count = torch::zeros({ current_batch_size, width * height }, device = device);
            std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt;

            // 生成路径
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

                // 更新已访问计数
                auto pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
                auto pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
                auto pixel_ids = pixel_x * height + pixel_y;

                auto current_visited = visited_count.clone();
                current_visited.scatter_add_(1, pixel_ids.unsqueeze(1),
                    torch::ones({ current_batch_size, 1 }, device = device));

                // 预测下一个位置

                auto [next_pos, new_hidden] = model->forward(
                    field, current_pos, current_visited, hidden_state
                );
                // ========== 添加子像素约束 ==========
               

                // 将连续坐标映射到最近的子像素中心点
                auto next_pos_subpixel = next_pos * subpixel_scale;

                // 找到每个点所属的基像素
                auto base_pixel_x = (next_pos_subpixel.index({ torch::indexing::Slice(), 0 }) / subpixel_scale).floor() * subpixel_scale;
                auto base_pixel_y = (next_pos_subpixel.index({ torch::indexing::Slice(), 1 }) / subpixel_scale).floor() * subpixel_scale;

                // 9个候选点的相对偏移
                torch::Tensor candidate_offsets = torch::tensor({
                    {1, 1}, {1, 0}, {1, 2},
                    {0, 1}, {0, 0}, {0, 2},
                    {2, 1}, {2, 0}, {2, 2}
                    }, torch::dtype(torch::kFloat).device(device));

                // 扩展维度以进行批量计算
                base_pixel_x = base_pixel_x.unsqueeze(1).expand({ current_batch_size, 9 });
                base_pixel_y = base_pixel_y.unsqueeze(1).expand({ current_batch_size, 9 });
                candidate_offsets = candidate_offsets.unsqueeze(0).expand({ current_batch_size, 9, 2 });

                // 计算所有候选点坐标
                auto candidate_x = base_pixel_x + candidate_offsets.index({ torch::indexing::Slice(), torch::indexing::Slice(), 0 });
                auto candidate_y = base_pixel_y + candidate_offsets.index({ torch::indexing::Slice(), torch::indexing::Slice(), 1 });

                // 计算到每个候选点的距离
                auto current_pos_expanded = current_pos.unsqueeze(1).expand({ current_batch_size, 9, 2 });
                auto next_pos_expanded = next_pos_subpixel.unsqueeze(1).expand({ current_batch_size, 9, 2 });

                auto candidate_points = torch::stack({ candidate_x, candidate_y }, 2);
                auto distances = (next_pos_expanded - candidate_points).norm(2, 2);

                // 选择最近的候选点
                auto min_indices = distances.argmin(1);
                auto selected_candidates = candidate_points.index({ torch::arange(current_batch_size), min_indices });

                // 转换回原始坐标尺度
                next_pos = selected_candidates / subpixel_scale;
                // ========== 子像素约束结束 ==========
               


                path.push_back(next_pos);
                visited_count = current_visited;
                hidden_state = new_hidden;
                /*if (i > 0 && i % 200 == 0) {
                    hidden_state = std::nullopt;
                    std::cout << "Reset hidden state at step " << i << std::endl;
                }*/
            }

            // 将路径转换为张量
            auto path_tensor = torch::stack(path, 1);

            // 计算损失
            auto [loss, components] = criterion(path_tensor, field, width, height);

            // 反向传播和优化
            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 0.5);
            optimizer.step();

            // 累加损失
            total_loss += loss.item<float>();
            for (const auto& [key, value] : components) {
                if (key != "total") {
                    loss_components[key] += value;
                }
            }
            batch_count++;
        }


        // 计算平均损失
        float avg_loss = total_loss / batch_count;
        for (auto& [key, value] : loss_components) {
            value /= batch_count;
        }

        // 调整学习率
        scheduler.step(avg_loss);

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            best_model = std::make_shared<PathPlannerModel>(*model); // 深拷贝
            std::cout << "New best model saved with loss: " << best_loss << std::endl;
        }

        // 打印训练进度
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
    //返回最佳的模型
    if (best_model) {
        std::cout << "Returning best model with loss: " << best_loss << std::endl;
        return best_model;
    }
    else {
        std::cout << "No valid model found, returning last model." << std::endl;
        return model;
    }
}





// 可视化结果
//void visualize_results(const std::shared_ptr<PathPlannerModel>& model, int width = 32, int height = 32) {
//    // 创建测试数据
//    FieldDataset dataset(1, width, height);
//    auto example = dataset.get(0);
//    auto field = example.data.unsqueeze(0); // 添加批次维度
//    auto start_pos = example.target.unsqueeze(0);
//
//    // 设备配置
//    torch::Device device(torch::kCPU);
//    if (torch::cuda::is_available()) {
//        device = torch::Device(torch::kCUDA);
//        field = field.to(device);
//        start_pos = start_pos.to(device);
//    }
//
//    // 切换到评估模式
//    model->eval();
//    int path_length = static_cast<int>(width * height * 0.5);
//
//    // 生成路径
//    std::vector<torch::Tensor> path;
//    path.push_back(start_pos);
//
//    auto visited_count = torch::zeros({ 1, width * height }, device = device);
//    std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt;
//
//    torch::NoGradGuard no_grad; // 禁用梯度计算
//    for (int i = 0; i < path_length - 1; ++i) {
//        auto current_pos = path.back();
//
//        // 更新已访问计数
//        auto pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
//        auto pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
//        auto pixel_ids = pixel_x * height + pixel_y;
//
//        auto current_visited = visited_count.clone();
//        current_visited.scatter_add_(1, pixel_ids.unsqueeze(1),
//            torch::ones({ 1, 1 }, device = device));
//
//        // 预测下一个位置
//        auto [next_pos, new_hidden] = model->forward(
//            field, current_pos, current_visited, hidden_state
//        );
//
//        path.push_back(next_pos);
//        visited_count = current_visited;
//        hidden_state = new_hidden;
//    }
//
//    // 转换为CPU并可视化
//    auto path_tensor = torch::stack(path, 1).cpu();
//    auto field_cpu = field.squeeze(0).cpu();
//    auto start_pos_cpu = start_pos.squeeze(0).cpu();
//
//    // 提取路径点
//    std::vector<double> path_x, path_y;
//    for (int i = 0; i < path_tensor.size(1); ++i) {
//        path_x.push_back(path_tensor[0][i][0].item<float>());
//        path_y.push_back(path_tensor[0][i][1].item<float>());
//    }
//
//    // 绘制场向量（稀疏显示）
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
//        //// 先设置属性映射
//        //std::map<std::string, std::string> quiver_props;
//        //quiver_props["color"] = "lightblue";
//        //quiver_props["linewidth"] = "0.5";
//
//        //// 传递属性映射
//        //plt::quiver(x, y, u, v, quiver_props);
//    }
//
//    std::map<std::string, std::string> quiver_props;
//    quiver_props["color"] = "lightblue";
//    quiver_props["linewidth"] = "0.5";
//
//    // 传递属性映射
//    plt::quiver(x, y, u, v, quiver_props);
//    //plt::quiver(x, y, u, v, { {"color", "lightblue"}, {"linewidth", 0.5} });
//
//    // 绘制路径
//
//    //plt::plot(path_x, path_y, "r-", { {"linewidth", 2} });
//    std::string p_for = "r-";
//    plt::plot(path_x, path_y, p_for);
//   
//
//    // 标记起点和终点
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

// 修改的visualize_results函数，从这里开始替换：

void visualize_results(const std::shared_ptr<PathPlannerModel>& model, int width = 32, int height = 32) {
    try {
        // 创建测试数据 - 这部分保持不变
        FieldDataset dataset(1, width, height);
        auto example = dataset.get(0);
        auto field = example.data.unsqueeze(0); // 添加批次维度
        auto start_pos = example.target.unsqueeze(0);

        // 设备配置 - 保持不变
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            field = field.to(device);
            start_pos = start_pos.to(device);
        }

        // 切换到评估模式 - 保持不变
        model->eval();
        int path_length = static_cast<int>(width * height * 8);

        // 生成路径 - 保持不变
        std::vector<torch::Tensor> path;
        path.push_back(start_pos);

        auto visited_count = torch::zeros({ 1, width * height }, device = device);
        std::optional<std::tuple<torch::Tensor, torch::Tensor>> hidden_state = std::nullopt;

        torch::NoGradGuard no_grad;
        for (int i = 0; i < path_length - 1; ++i) {
            auto current_pos = path.back();

            // 更新已访问计数
            auto pixel_x = current_pos.index({ torch::indexing::Slice(), 0 }).clamp(0, width - 1).to(torch::kLong);
            auto pixel_y = current_pos.index({ torch::indexing::Slice(), 1 }).clamp(0, height - 1).to(torch::kLong);
            auto pixel_ids = pixel_x * height + pixel_y;

            auto current_visited = visited_count.clone();
            current_visited.scatter_add_(1, pixel_ids.unsqueeze(1),
                torch::ones({ 1, 1 }, device = device));

            // 预测下一个位置
            auto [next_pos, new_hidden] = model->forward(
                field, current_pos, current_visited, hidden_state
            );

            path.push_back(next_pos);
            visited_count = current_visited;
            hidden_state = new_hidden;
        }

        // ========== 从这里开始替换为OpenCV版本 ==========

        // 转换为CPU
        auto path_tensor = torch::stack(path, 1).cpu();
        auto field_cpu = field.squeeze(0).cpu();

        // 创建OpenCV图像（放大显示）
        int scale = 60; // 放大倍数
        cv::Mat image = cv::Mat::ones(height * scale, width * scale, CV_8UC3) * 255;

        // 绘制网格
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

        // 绘制向量场
        int step = std::max(1, width / 10);
        for (int i = 0; i < height; i += step) {
            for (int j = 0; j < width; j += step) {
                float u = field_cpu[0][i][j].item<float>();
                float v = field_cpu[1][i][j].item<float>();

                // 计算箭头起点和终点
                cv::Point start(j * scale + scale / 2, i * scale + scale / 2);
                cv::Point end(start.x + u * scale * 0.4f, start.y + v * scale * 0.4f);

                // 绘制箭头
                cv::arrowedLine(image, start, end, cv::Scalar(100, 100, 255), 2, 8, 0, 0.3);
            }
        }

        // 提取路径点
        std::vector<cv::Point> path_points;
        for (int i = 0; i < path_tensor.size(1); ++i) {
            float x = path_tensor[0][i][0].item<float>();
            float y = path_tensor[0][i][1].item<float>();
            path_points.push_back(cv::Point(x * scale, y * scale));
        }

        // 绘制路径线
        for (size_t i = 1; i < path_points.size(); ++i) {
            cv::line(image, path_points[i - 1], path_points[i], cv::Scalar(255, 0, 0), 2);
        }
        //==========================主要修改为绘制路径而不是点============================
        // 绘制路径点
       /* for (const auto& point : path_points) {
            cv::circle(image, point, 3, cv::Scalar(0, 0, 255), -1);
        }*/
        cv::Mat path_image = image.clone(); // 创建路径的单独图像

        // 使用更粗的线条绘制主要路径
        for (size_t i = 1; i < path_points.size(); ++i) {
            // 根据路径段的长度调整线条粗细（较长的段用较粗的线）
            double segment_length = cv::norm(path_points[i] - path_points[i - 1]);
            int thickness = std::max(2, static_cast<int>(segment_length / 10));

            // 使用渐变色：从绿色（起点）到蓝色（中间）到红色（终点）
            double progress = static_cast<double>(i) / path_points.size();
            cv::Scalar color;
            if (progress < 0.33) {
                // 绿色到蓝色的渐变
                color = cv::Scalar(255 * (0.33 - progress) * 3, 255, 0);
            }
            else if (progress < 0.66) {
                // 蓝色到紫色的渐变
                color = cv::Scalar(0, 255 * (0.66 - progress) * 3, 255);
            }
            else {
                // 紫色到红色的渐变
                color = cv::Scalar(255 * (progress - 0.66) * 3, 0, 255);
            }

            cv::line(path_image, path_points[i - 1], path_points[i], color, thickness);
            cv::circle(path_image, path_points[i], 2, cv::Scalar(0, 0, 255), -1); // 红色小圆点
            // 每隔一定距离绘制方向箭头
            static double accumulated_distance = 0;
            accumulated_distance += segment_length;

            // 设置箭头间隔（例如每50像素绘制一个箭头）
            const double arrow_interval = 50.0;

            if (accumulated_distance >= arrow_interval) {
                // 计算箭头位置（在线段中点或根据比例）
                double arrow_ratio = 0.5; // 箭头在线段上的位置比例（0-1之间）
                cv::Point arrow_pos = path_points[i - 1] +
                    cv::Point((path_points[i] - path_points[i - 1]) * arrow_ratio);

                // 计算线段方向向量
                cv::Point direction = path_points[i] - path_points[i - 1];
                double angle = atan2(direction.y, direction.x) * 180 / CV_PI;

                // 绘制紫色小箭头
                cv::Scalar arrow_color(255, 0, 255); // 紫色(BGR格式)
                int arrow_length = 15; // 箭头长度
                int arrow_thickness = 5; // 箭头粗细

                // 使用arrowedLine函数绘制箭头
                cv::arrowedLine(path_image,
                    arrow_pos - cv::Point(arrow_length / 2 * cos(angle * CV_PI / 180),
                        arrow_length / 2 * sin(angle * CV_PI / 180)),
                    arrow_pos + cv::Point(arrow_length / 2 * cos(angle * CV_PI / 180),
                        arrow_length / 2 * sin(angle * CV_PI / 180)),
                    arrow_color, arrow_thickness, 8, 0, 0.3);

                // 重置累积距离
                accumulated_distance = 0;

            }

        }

        for (size_t i = 0; i < path_points.size(); ++i) {
            // 使用半透明的红色圆点，这样重合的点会显示得更深
            cv::Scalar point_color(0, 0, 255); // 红色
            int point_radius = 10;

            // 在最终图像上绘制点（在混合之后）
            cv::circle(image, path_points[i], point_radius, point_color, -1);

            // 如果点很密集，可以每隔几个点绘制一个，避免太乱
            // if (i % 5 == 0) { // 每5个点画一个
            //     cv::circle(image, path_points[i], point_radius, point_color, -1);
            // }
        }

        double alpha = 0.7; // 路径透明度
        cv::addWeighted(path_image, alpha, image, 1 - alpha, 0, image);
        //绘制箭头表示方向




        // 标记起点和终点
        if (!path_points.empty()) {
            // 起点（绿色）
            cv::circle(image, path_points.front(), 8, cv::Scalar(0, 255, 0), -1);
            cv::putText(image, "Start", path_points.front() + cv::Point(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // 终点（红色）
            cv::circle(image, path_points.back(), 8, cv::Scalar(0, 0, 255), -1);
            cv::putText(image, "End", path_points.back() + cv::Point(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        // 添加标题和信息
        cv::putText(image, "Path Planning Visualization", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        cv::putText(image, "Blue: Path, Pink: Vector Field", cv::Point(10, height * scale - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        for (cv::Point i: path_points) {
            std::cout << i << std::endl;
        }
    
        // 显示图像
        cv::imshow("Path Planning Result", image);
        std::cout << "Press any key to close the window..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();

        // 保存图像
        cv::imwrite("path_planning_result.png", image);
        std::cout << "Image saved as path_planning_result.png" << std::endl;

    }
    

    catch (const std::exception& e) {
        std::cout << "Error in visualization: " << e.what() << std::endl;
    }
}

int main() {
    // 设置随机种子
    torch::manual_seed(42);

    // 训练模型
    int width = 8, height = 8;
    std::cout << "Training model for grid size " << width << "x" << height << "..." << std::endl;
    auto model = train_model(50, 16, width, height);

    // 可视化结果
    std::cout << "Visualizing results..." << std::endl;
    visualize_results(model, width, height);

    return 0;
}
