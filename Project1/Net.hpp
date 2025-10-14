#pragma once
#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>
class InputProcessor{
public:
    InputProcessor() {};
    // 输入: 8x8区域掩码 + 8x8x2矢量场
    torch::Tensor process(const torch::Tensor& region_mask,
        const torch::Tensor& vector_field);
    torch::Tensor upsample_vector_field_only(const torch::Tensor& vector_field);

    torch::Tensor upsample_3x3(const torch::Tensor& mask);
    torch::Tensor upsample_vector_field(const torch::Tensor& vector_field);
    torch::Tensor create_position_grid(int height, int width);
    torch::Tensor upsample_3x3_smooth(const torch::Tensor& field);
};

class CNNEncoder : public torch::nn::Module {
private:
    // 多尺度特征提取
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };
    torch::nn::MaxPool2d pool{ nullptr };
public:
    CNNEncoder();
    torch::Tensor forward(const torch::Tensor& x);
    //std::vector<torch::Tensor> forward_with_features(const torch::Tensor& x);
};

class AttentionLayer : public torch::nn::Module {
private:
    torch::nn::Linear 
        query_proj{ nullptr }, 
        key_proj{ nullptr }, 
        value_proj{ nullptr };
    int hidden_size;
public:
    AttentionLayer(int hidden_size);
    torch::Tensor forward(const torch::Tensor& lstm_hidden,
        const torch::Tensor& cnn_features);

};

class LSTMPathDecoder : public torch::nn::Module {
private:
    torch::nn::LSTM lstm{ nullptr };// LSTM主体：处理序列依赖，输出隐状态
    std::shared_ptr<AttentionLayer> attention{ nullptr };  // 使用shared_ptr// 注意力层：聚焦CNN特征中的关键区域
    torch::nn::Linear point_embedding{ nullptr };// 点嵌入层：将2D坐标转为高维特征
    torch::nn::Linear output_layer{ nullptr };// 输出层：预测下一个点的概率分布
    torch::nn::Linear coverage_embedding{ nullptr };// 覆盖嵌入层：将覆盖图转为高维特征
    torch::nn::Linear feature_projection{ nullptr };// 特征投影层：匹配CNN特征与注意力维度
    int hidden_size;// LSTM隐状态维度（默认512）
    int m_layer_num;
public:
    int get_m_layer_num() {
        return m_layer_num;
    }
    int get_hidden_size();
    LSTMPathDecoder(int hidden_size = 512,int layer_num = 2) ;

    struct DecoderState {
        torch::Tensor hidden_state; //LSTM隐状态[2, batch, hidden_size]（2层）
        torch::Tensor cell_state;// LSTM细胞状态 [2, batch, hidden_size]
        torch::Tensor coverage_map;  // 当前覆盖状态 [batch, 24, 24]
        torch::Tensor last_point;    // 上一个点坐标 [batch, 2]
        int step_count = 0;// 当前步数（用于控制最大长度）

        // 构造函数
        DecoderState(const torch::Tensor& region_mask, int batch_size = 1, int hidden_size = 512, int num_layers = 2) {
            // 初始化隐藏状态和细胞状态
             // 从region_mask获取尺寸信息
            int64_t original_height = region_mask.size(0);
            int64_t original_width = region_mask.size(1);
            int64_t highres_height = original_height * 3;  // 九宫格上采样
            int64_t highres_width = original_width * 3;

            hidden_state = torch::zeros({ num_layers, batch_size, hidden_size });  // 2层LSTM
            cell_state = torch::zeros({ num_layers, batch_size, hidden_size });

            coverage_map = initialize_coverage_from_mask(region_mask, batch_size);//从输入的区域掩码初始化掩码

            last_point = torch::full({ batch_size, 2 },1.0f);  // 初始点设为(0.5,0.5)
            step_count = 0;
        }
    
        // 根据region_mask初始化覆盖图
        torch::Tensor initialize_coverage_from_mask(const torch::Tensor& region_mask, int batch_size) {
            int64_t h = region_mask.size(0);
            int64_t w = region_mask.size(1);
            int64_t highres_h = h * 3;
            int64_t highres_w = w * 3;

            auto coverage = torch::zeros({ batch_size, highres_h, highres_w });

            // 如果region_mask有值，可以用来初始化覆盖状态
            // 例如：非打印区域初始化为已覆盖
            if (batch_size == 1) {
                for (int i = 0; i < h; ++i) {
                    for (int j = 0; j < w; ++j) {
                        if (region_mask[i][j].item<float>() < 0.5f) {
                            // 非打印区域，标记为已覆盖
                            coverage.index_put_(
                                { torch::indexing::Slice(),
                                 torch::indexing::Slice(i * 3, i * 3 + 3),
                                 torch::indexing::Slice(j * 3, j * 3 + 3) },
                                1.0f
                            );
                        }
                    }
                }
            }

            return coverage;//batch_size,24,24
        }
    };
    static std::vector<DecoderState> initialize_batch_states(
        const std::vector<torch::Tensor>& region_masks,
        torch::Device device,
        int hidden_size ,
        int layer_num )
     {
        
        std::vector<DecoderState> states;
        int batch_size = region_masks.size();
        //std::cout << "initialize_batch_states 接收的设备: " << device << std::endl;
        // 为批量中的每个样本创建独立的状态
        for (int i = 0; i < batch_size; ++i) {
            DecoderState state(region_masks[i], 1, hidden_size,layer_num);  // 每个样本batch_size=1

            // 移动到设备
            state.coverage_map = state.coverage_map.to(device);
            state.last_point = state.last_point.to(device);
            state.hidden_state = state.hidden_state.to(device);
            state.cell_state = state.cell_state.to(device);
            //std::cout<<state.last_point<<std::endl;


            states.push_back(state);
        }

        return states;
    }
    std::pair<torch::Tensor, DecoderState> forward(
        const torch::Tensor& cnn_features,   // [batch, 36, 256]
        DecoderState state,                  // 解码器状态
        const torch::Tensor& vector_field    // [batch, 24, 24, 2] 矢量场
    );
//private:
     torch::Tensor create_coverage_mask(const torch::Tensor& coverage_map);
    // 辅助函数：将点索引转换为坐标
    static torch::Tensor idx_to_coordinate(const torch::Tensor& indices, int grid_size = 24);
    // 辅助函数：更新覆盖状态
    static torch::Tensor update_coverage(const torch::Tensor& coverage_map,
        const torch::Tensor& point_coords);
    
};

//模拟温度退火的softmax
class ImprovedSampler {
private:
    torch::Device device;
    float initial_temperature;
    float min_temperature;
    float decay_rate;
    int training_step;

public:
    ImprovedSampler(torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, float init_temp = 1.0f,
        float min_temp = 0.1f, float decay = 0.9995f) : device(device), initial_temperature(init_temp),
        min_temperature(min_temp), decay_rate(decay), training_step(0) {
    }
    std::pair<torch::Tensor, torch::Tensor> sample_with_temperature_st(
        const torch::Tensor& point_logits);
    // 方法1：温度退火的Gumbel-Softmax（推荐）
    torch::Tensor sample_with_temperature(const torch::Tensor& point_logits);
private:
public:
    
    float get_current_temperature() {
        // 指数退火：随着训练进行逐渐降低温度
        float temp = initial_temperature * std::pow(decay_rate, training_step);
        return std::max(temp, min_temperature);
    }
    // 将概率分布转换为连续坐标
    torch::Tensor probability_to_continuous_coordinates(const torch::Tensor& probs, int grid_size);
    
};

class PathPlanningTrainer {
public:
    torch::Device device;
    CNNEncoder encoder;
    LSTMPathDecoder decoder;
    InputProcessor input_processor; 
    //torch::optim::Adam optimizer;
    std::unique_ptr<torch::optim::Adam> optimizer;
    ImprovedSampler sampler;
    

    // 训练参数
    float lambda_coverage = 0.8;
    float lambda_overlap = 0.9;
    float lambda_smoothness = 1.0;
    float lambda_alignment = 1.0;
    int max_path_length = 300;
    int batch_size ;

public:
    PathPlanningTrainer(torch::Device device = torch::kCUDA, float learning_rate = 1e-4);
    torch::Tensor train_step(std::vector<torch::Tensor> region_masks, std::vector<torch::Tensor> vector_fields
        ,int samples_num = 4, int width=8, int height=8);

    bool is_coverage_complete(const torch::Tensor& coverage_map,
        const torch::Tensor& region_mask);
    torch::Tensor update_coverage(const torch::Tensor& coverage_map,
        const torch::Tensor& point_coords);
    std::vector<torch::Tensor> get_all_parameters();
    void print_gradient_info(const std::vector<torch::Tensor>& parameters);
};

class MultiObjectiveLoss {

private:
        torch::Device device;
        float lambda_coverage;
        float lambda_overlap;
        float lambda_smoothness;
        float lambda_alignment;
        

public:
    MultiObjectiveLoss(torch::Device device = torch::kCUDA,
        float coverage_weight = 1.0f,
        float overlap_weight = 0.5f,
        float smoothness_weight = 0.3f,
        float alignment_weight = 0.2f) : device(device), lambda_coverage(coverage_weight),
        lambda_overlap(overlap_weight), lambda_smoothness(smoothness_weight),
        lambda_alignment(alignment_weight) {
    }
    // 批量损失计算接口
    torch::Tensor compute_batch_loss(
        const std::vector<std::vector<torch::Tensor>>& batch_paths,      // [batch_size][path_length, 2]
        const std::vector<std::vector<torch::Tensor>>& batch_coverages,  // [batch_size][path_length, 24, 24]
        const torch::Tensor& vector_fields,                              // [batch_size, 24, 24, 2]
        const std::vector<torch::Tensor>& region_masks);
    // 单个样本损失计算
    torch::Tensor compute_sample_loss(
        const std::vector<torch::Tensor>& path_points,      // [path_length, 2]
        const std::vector<torch::Tensor>& coverage_maps,    // [path_length, 24, 24]  
        const torch::Tensor& vector_field,                  // [24, 24, 2]
        const torch::Tensor& region_mask);
private:
    // 1. 覆盖损失 - 计算未覆盖区域的比例
    torch::Tensor compute_coverage_loss(const torch::Tensor& final_coverage,
        const torch::Tensor& region_mask);
    // 2. 重叠损失 - 计算非相邻点之间的重叠
    torch::Tensor compute_overlap_loss(const std::vector<torch::Tensor>& path_points,
        const std::vector<torch::Tensor>& coverage_maps);

    // 3. 平滑损失 - 计算路径转折角度
    torch::Tensor compute_smoothness_loss(const std::vector<torch::Tensor>& path_points);
    // 4. 场对齐损失 - 计算路径段与矢量场的对齐程度
    torch::Tensor compute_alignment_loss(const std::vector<torch::Tensor>&path_points,
        const torch::Tensor & vector_field);
    // 辅助函数：上采样掩码
    torch::Tensor upsample_mask_3x3(const torch::Tensor& mask);
    // 辅助函数：获取点的覆盖区域
    torch::Tensor get_point_coverage(const torch::Tensor& point);
    // 辅助函数：双线性采样矢量场
    torch::Tensor bilinear_sample_vector_field(const torch::Tensor& vector_field,
        const torch::Tensor& point);

};
//模拟温度退火的softmax
//class ImprovedSampler {
//private:
//    torch::Device device;
//    float initial_temperature;
//    float min_temperature;
//    float decay_rate;
//    int training_step;
//
//public:
//    ImprovedSampler(torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, float init_temp = 1.0f,
//        float min_temp = 0.1f, float decay = 0.9995f) : device(device), initial_temperature(init_temp),
//        min_temperature(min_temp), decay_rate(decay), training_step(0) {}
//    // 方法1：温度退火的Gumbel-Softmax（推荐）
//    torch::Tensor sample_with_temperature(const torch::Tensor& point_logits);
//private:
//    float get_current_temperature() {
//        // 指数退火：随着训练进行逐渐降低温度
//        float temp = initial_temperature * std::pow(decay_rate, training_step);
//        return std::max(temp, min_temperature);
//    }
//};