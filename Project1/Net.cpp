#include "Net.hpp"
#include "FieldDataset.hpp"
//class InputProcessor
torch::Tensor InputProcessor::process(const torch::Tensor& region_mask,
    const torch::Tensor& vector_field) {//注意掩码形状与输入场形状要一致
    torch::Device device = region_mask.device();  // 获取输入设备（如GPU）
    // 1. 区域掩码上采样到24x24 (九宫格细化)
    auto highres_mask = upsample_3x3(region_mask);

    // 2. 矢量场上采样到24x24x2
    auto highres_field = upsample_vector_field(vector_field);

    int64_t original_height = vector_field.size(0);
    int64_t original_width = vector_field.size(1);
    // 3. 创建位置编码网格
    auto grid = create_position_grid(original_height*3, original_width*3).to(device);//应该与传入的两个张量一致，

    // 4. 拼接所有特征: [掩码, 场方向, 位置] → 24x24x5
    return torch::cat({ highres_mask, highres_field, grid }, -1);
}
torch::Tensor InputProcessor::upsample_3x3(const torch::Tensor& mask) {
    // 输入: 8x8, 输出: 24x24
    TORCH_CHECK(mask.dim() == 2, "输入掩码必须是2D张量，实际维度: ", mask.dim());
    // 从输入mask获取原始维度 [H, W]
    int64_t original_height = mask.size(0);
    int64_t original_width = mask.size(1);//8x8为例
    //std::cout << original_height << original_width << std::endl;
    // 计算3倍上采样后的目标维度
    int64_t target_height = original_height * 3;
    int64_t target_width = original_width * 3;

    auto options = torch::TensorOptions().dtype(mask.dtype()).device(mask.device());
    auto result = torch::zeros({ target_height, target_width }, options);

    for (int i = 0; i < original_height; ++i) {
        for (int j = 0; j < original_width; ++j) {
            float value = mask[i][j].item<float>();
            // 将每个8x8的像素扩展为3x3块
            result.index_put_({ torch::indexing::Slice(i * 3, i * 3 + 3),
                             torch::indexing::Slice(j * 3, j * 3 + 3) }, value);
        }
    }
    return result.unsqueeze(-1);  // 24x24x1
}
torch::Tensor InputProcessor::upsample_vector_field(const torch::Tensor& vector_field) {
    // 输入: 8x8x2, 输出: 24x24x2
    TORCH_CHECK(vector_field.dim() == 3,
        "矢量场必须是3D张量 [H, W, 2]，实际维度: ", vector_field.dim());
    TORCH_CHECK(vector_field.size(2) == 2,
        "矢量场最后一维必须为2（x和y分量），实际尺寸: ", vector_field.size(2));
    

    // 方法1: 简单的重复插值
    auto field_x = vector_field.index({ "...", 0 });  // 8x8
    auto field_y = vector_field.index({ "...", 1 });  // 8x8

    // 上采样x分量
    auto upsampled_x = upsample_3x3_smooth(field_x);
    auto upsampled_y = upsample_3x3_smooth(field_y);

    // 合并为24x24x2
    return torch::stack({ upsampled_x, upsampled_y }, -1);
}
torch::Tensor InputProcessor::upsample_3x3_smooth(const torch::Tensor& field) {

    TORCH_CHECK(field.dim() == 2, "输入必须是2D张量，实际维度: ", field.dim());

    // 从输入动态获取原始维度
    int64_t original_height = field.size(0);
    int64_t original_width = field.size(1);

    // 计算3倍上采样后的目标维度
    int64_t target_height = original_height * 3;
    int64_t target_width = original_width * 3;

    // 创建与输入同类型、同设备的输出张量
    auto options = torch::TensorOptions().dtype(field.dtype()).device(field.device());
    auto result = torch::zeros({ target_height, target_width }, options);

    for (int i = 0; i < original_height; ++i) {
        for (int j = 0; j < original_width; ++j) {
            float value = field[i][j].item<float>();

            // 获取相邻值用于插值
            float left = (j > 0) ? field[i][j - 1].item<float>() : value;
            float right = (j < original_width - 1) ? field[i][j + 1].item<float>() : value;
            float top = (i > 0) ? field[i - 1][j].item<float>() : value;
            float bottom = (i < original_height - 1) ? field[i + 1][j].item<float>() : value;

            // 3x3块内的插值权重
            for (int di = 0; di < 3; ++di) {
                for (int dj = 0; dj < 3; ++dj) {
                    float weight_x = 1.0f - std::abs(dj - 1.0f) / 1.0f;  // 水平权重
                    float weight_y = 1.0f - std::abs(di - 1.0f) / 1.0f;  // 垂直权重

                    float interpolated = value;
                    if (dj == 0 && j > 0) {  // 左边界，向左插值
                        interpolated = value * 0.7f + left * 0.3f;
                    }
                    else if (dj == 2 && j < original_width - 1) {  // 右边界，向右插值
                        interpolated = value * 0.7f + right * 0.3f;
                    }
                    else if (di == 0 && i > 0) {  // 上边界，向上插值
                        interpolated = value * 0.7f + top * 0.3f;
                    }
                    else if (di == 2 && i < original_height - 1) {  // 下边界，向下插值
                        interpolated = value * 0.7f + bottom * 0.3f;
                    }

                    result[i * 3 + di][j * 3 + dj] = interpolated;
                }
            }
        }
    }
    return result;
}
torch::Tensor InputProcessor::create_position_grid(int height, int width) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto grid = torch::zeros({ height, width, 2 }, options);

    // 创建归一化坐标 [-1, 1]
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float norm_x = (2.0f * j / (width - 1)) - 1.0f;  // x从-1到1
            float norm_y = (2.0f * i / (height - 1)) - 1.0f; // y从-1到1

            grid[i][j][0] = norm_x;
            grid[i][j][1] = norm_y;
        }
    }
    return grid;
}
torch::Tensor InputProcessor::upsample_vector_field_only(const torch::Tensor& vector_field) {
    return upsample_vector_field(vector_field);
}

//class CNNEncoder
CNNEncoder::CNNEncoder() {
    // 第一层: 提取局部特征
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 64, 3).stride(1).padding(1)));  // 输入5通道, 输出64通道
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

    // 第二层: 中等尺度特征
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));

    // 第三层: 全局特征
    conv3 = register_module("conv3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));

    pool = register_module("pool", torch::nn::MaxPool2d(2));
}
torch::Tensor CNNEncoder::forward(const torch::Tensor& x) {

    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor [batch, height, width, channels]");
    TORCH_CHECK(x.size(1) == 24 && x.size(2) ==
        24 && x.size(3) == 5,
        "Input must be [batch, 24, 24, 5]");
    // 将格式从 [batch, height, width, channels] 转换为 [batch, channels, height, width]
    auto x_permuted = x.permute({ 0, 3, 1, 2 });  // 现在形状是 [batch, 5, 24, 24]

    // 第一层卷积: [batch, 5, 24, 24] → [batch, 64, 24, 24]
    auto h = torch::relu(bn1(conv1(x_permuted)));

    // 第一次池化: [batch, 64, 24, 24] → [batch, 128, 12, 12]
    h = pool(torch::relu(bn2(conv2(h))));

    // 第二次池化: [batch, 128, 12, 12] → [batch, 256, 6, 6]
    h = pool(torch::relu(bn3(conv3(h))));

    // 展平特征图: [batch, 256, 6, 6] → [batch, 36, 256]
    // 将空间维度展平为序列，保持通道维度作为特征
    auto batch_size = h.size(0);
    h = h.permute({ 0, 2, 3, 1 });  // [batch, 6, 6, 256]
    h = h.reshape({ batch_size, 36, 256 });  // [batch, 36, 256]

    return h;
}
//std::vector<torch::Tensor> CNNEncoder::forward_with_features(const torch::Tensor& x) {
//
//    auto x_permuted = x.permute({ 0, 3, 1, 2 });
//
//    std::vector<torch::Tensor> features;
//
//    auto h1 = torch::relu(bn1(conv1(x_permuted)));
//    features.push_back(h1);  // [batch, 64, 24, 24]
//
//    auto h2 = torch::relu(bn2(conv2(h1)));
//    auto h2_pooled = pool(h2);
//    features.push_back(h2_pooled);  // [batch, 128, 12, 12]
//
//    auto h3 = torch::relu(bn3(conv3(h2_pooled)));
//    auto h3_pooled = pool(h3);
//    features.push_back(h3_pooled);  // [batch, 256, 6, 6]
//
//    // 最终输出
//    auto batch_size = h3_pooled.size(0);
//    auto output = h3_pooled.permute({ 0, 2, 3, 1 }).reshape({ batch_size, 36, 256 });
//    features.push_back(output);
//
//    return features;
//}

//class AttentionLayer
AttentionLayer::AttentionLayer(int hidden_size): hidden_size(hidden_size) {//将层注册到module中
        query_proj = register_module("query_proj",
            torch::nn::Linear(hidden_size, hidden_size));//输入维度=输出维度
        key_proj = register_module("key_proj",
            torch::nn::Linear(256, hidden_size));  // CNN特征维度256,输出为hidden_size一个是键值
        value_proj = register_module("value_proj",
            torch::nn::Linear(256, hidden_size));//一个是值
        //初始化

        torch::nn::init::xavier_uniform_(query_proj->weight);
        torch::nn::init::zeros_(query_proj->bias);
        torch::nn::init::xavier_uniform_(key_proj->weight);
        torch::nn::init::zeros_(key_proj->bias);
        torch::nn::init::xavier_uniform_(value_proj->weight);
        torch::nn::init::zeros_(value_proj->bias);
    }
torch::Tensor AttentionLayer::forward(const torch::Tensor& lstm_hidden,
    const torch::Tensor& cnn_features) {
    // lstm_hidden: [batch, hidden_size]
    // cnn_features: [batch, 36, 256]

    auto query = torch::leaky_relu(query_proj(lstm_hidden),0.1).unsqueeze(1);  // [batch, 1, hidden]只有一个查询量，lstm中来
    auto keys = torch::leaky_relu(key_proj(cnn_features),0.1);                 // [batch, 36, hidden]
    auto values = torch::leaky_relu(value_proj(cnn_features),0.1);             // [batch, 36, hidden]

    // 计算注意力权重
    auto scores = torch::bmm(query, keys.transpose(1, 2));  // [batch, 1, 36]
    auto attn_weights = torch::softmax(scores, 2);

    // 加权求和
    // 加权求和：[batch, 1, 36] × [batch, 36, hidden] → [batch, 1, hidden] → [batch, hidden]
    auto context = torch::bmm(attn_weights, values).squeeze(1);  // [batch, hidden]
    return context;
}

//class LSTMPathDecoder
LSTMPathDecoder::LSTMPathDecoder(int hidden_size,int layer_num) : hidden_size(hidden_size) {
    // 正确的输入尺寸计算：
    // point_embedding输出: 32维
    // coverage_embedding输出: 128维  
    // attention上下文: hidden_size维
    // 总输入尺寸: 32 + 128 + hidden_size
    int input_size = 32 + 128 + hidden_size;
    m_layer_num = layer_num;
    // 2层LSTM，batch_first=true
    lstm = register_module("lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
            .num_layers(m_layer_num)
            .batch_first(true)
            /*.dropout(0.1)*/));

    // 使用make_shared正确初始化AttentionLayer
    attention = std::make_shared<AttentionLayer>(hidden_size);
    register_module("attention", attention);

    point_embedding = register_module("point_embedding",
        torch::nn::Linear(2, 32));  // 2D坐标嵌入到32维

    coverage_embedding = register_module("coverage_embedding",
        torch::nn::Linear(576, 128));  // 覆盖状态嵌入(24x24=576)

    output_layer = register_module("output_layer",
        torch::nn::Linear(hidden_size, 576));  // 输出576个点的概率

    // 可选的：特征投影层，确保CNN特征维度匹配
    /*feature_projection = register_module("feature_projection",
        torch::nn::Linear(256, 256));*/
    //初始化权重
    torch::nn::init::xavier_uniform_(point_embedding->weight);
    torch::nn::init::zeros_(point_embedding->bias);

    torch::nn::init::xavier_uniform_(coverage_embedding->weight);
    torch::nn::init::zeros_(coverage_embedding->bias);

    for (auto& param : lstm->named_parameters()) {
        if (param.key().find("weight") != std::string::npos) {
            torch::nn::init::orthogonal_(param.value());
        }
        else {
            torch::nn::init::zeros_(param.value());
        }
    }
}
std::pair<torch::Tensor, LSTMPathDecoder::DecoderState> LSTMPathDecoder::forward(
    const torch::Tensor& cnn_features,   // [batch, 36, 256]
    DecoderState state,                  // 解码器状态
    const torch::Tensor& vector_field    // [batch, 24, 24, 2] 矢量场,但是没有用
) {
    auto batch_size = cnn_features.size(0);
   // std::cout <<"LSTMPathDecoder_batch_size:" << batch_size << std::endl;
    // 1. 准备LSTM输入特征

    // 点坐标嵌入: [batch, 2] -> [batch, 32]
    //std::cout << "state.last_point" << std::endl;
   // std::cout << state.last_point << std::endl;
    //std::cout << state.last_point.device() << std::endl;
    auto point_embed = torch::leaky_relu(point_embedding(state.last_point), 0.1);
   // std::cout <<"point_embed:" << point_embed.sizes() << std::endl;
    // 覆盖状态嵌入: [batch, 24, 24] -> [batch, 576] -> [batch, 128]
    auto coverage_flat = state.coverage_map.reshape({ batch_size, -1 });  // [batch, 576]
    auto coverage_embed = torch::leaky_relu(coverage_embedding(coverage_flat), 0.1); // [batch, 128]
    //std::cout<<"coverage_embed.sizes():" << coverage_embed.sizes() << std::endl;
    //std::cout << state.hidden_state[-1].sizes() << std::endl;

    // 2. 计算注意力上下文
    // 使用LSTM最后一层的最后一个时间步的隐藏状态
    auto last_hidden = state.hidden_state[-1];  // [batch, hidden_size]
    
    auto context = attention->forward(last_hidden, cnn_features);  // [batch, hidden_size]

    // 3. 拼接输入特征: [batch, 32+128+hidden_size]
    auto lstm_input = torch::cat({ point_embed, coverage_embed, context }, 1);
    lstm_input = lstm_input.unsqueeze(1);  // [batch, 1, input_size]

    // 4. LSTM前向传播
    auto lstm_output = lstm->forward(lstm_input,
        std::pair<torch::Tensor, torch::Tensor>{state.hidden_state, state.cell_state });

    auto lstm_out = std::get<0>(lstm_output);  // [batch, 1, hidden_size]
    auto new_states = std::get<1>(lstm_output);

    // 5. 输出层: 预测下一个点的概率分布
    auto lstm_hidden = lstm_out.squeeze(1);  // [batch, hidden_size]
    auto point_logits = output_layer(lstm_hidden);  // [batch, 576]

    // 6.1 应用覆盖掩码
    auto coverage_mask = create_coverage_mask(state.coverage_map);  // [batch, 576]
    point_logits = point_logits - 1* coverage_mask;// 已覆盖区域的点几乎不会被选中(惩罚可能有点多！！！！！)
    // 6.2限制到该点的7*7邻域区域概率大
    auto last_x = state.last_point.index({ torch::indexing::Slice(), 0 }).to(torch::kInt);  // [batch]，x坐标（列）
    auto last_y = state.last_point.index({ torch::indexing::Slice(), 1 }).to(torch::kInt);  // [batch]，y坐标（行）
    auto range_mask = torch::ones({ batch_size, 576 }, point_logits.device());
    for (int b = 0; b < batch_size; ++b) {
        int x = last_x[b].item<int>();  // 当前样本上一点的x坐标
        int y = last_y[b].item<int>();  // 当前样本上一点的y坐标

        // 计算7×7区域的边界（确保在0~23范围内）
        int x_min = std::max(0, x - 3);    // x方向左边界（最小0）
        int x_max = std::min(23, x + 3);   // x方向右边界（最大23）
        int y_min = std::max(0, y - 3);    // y方向上边界（最小0）
        int y_max = std::min(23, y + 3);   // y方向下边界（最大23）

        // 标记7×7区域内的点：掩码设为0（不惩罚）
        for (int yi = y_min; yi <= y_max; ++yi) {
            for (int xi = x_min; xi <= x_max; ++xi) {
                int idx = yi * 24 + xi;  // 点的索引（0~575）
                range_mask[b][idx] = 0.0f;  // 7×7区域内的点不惩罚
            }
        }
    }
    point_logits = point_logits - 1 * range_mask;//先设置为同样的概率

    // 7. 更新解码器状态
    torch::Tensor sam = state.coverage_map[0];//24x24
    DecoderState new_state(sam, batch_size);
    new_state.hidden_state = std::get<0>(new_states);
    new_state.cell_state = std::get<1>(new_states);

    new_state.coverage_map = state.coverage_map;  // 在外部更新
    new_state.last_point = state.last_point;      // 在外部更新 

    new_state.step_count = state.step_count + 1;

    return { point_logits, new_state };
}
torch::Tensor LSTMPathDecoder::create_coverage_mask(const torch::Tensor& coverage_map) {
    //// coverage_map: [batch, 24, 24]
    //auto batch_size = coverage_map.size(0);

    //// 创建掩码：如果点对应的3x3区域已完全覆盖，则掩码为1
    //auto mask = torch::zeros({ batch_size, 576 });//这里也需要改

    //for (int b = 0; b < batch_size; ++b) {
    //    for (int i = 0; i < 24; ++i) {
    //        for (int j = 0; j < 24; ++j) {
    //            int idx = i * 24 + j;

    //            // 检查3x3区域是否已完全覆盖
    //            bool fully_covered = true;
    //            for (int di = -1; di <= 1; ++di) {
    //                for (int dj = -1; dj <= 1; ++dj) {
    //                    int ni = i + di, nj = j + dj;
    //                    if (ni >= 0 && ni < 24 && nj >= 0 && nj < 24) {
    //                        if (coverage_map[b][ni][nj].item<float>() < 0.9f) {
    //                            fully_covered = false;
    //                            break;
    //                        }
    //                    }
    //                    else {
    //                        fully_covered = false;  // 边界区域不算完全覆盖
    //                    }
    //                }
    //                if (!fully_covered) break;
    //            }

    //            if (fully_covered) {
    //                mask[b][idx] = 1.0f;
    //            }
    //        }
    //    }
    //}

    //return mask;
    auto batch_size = coverage_map.size(0);
    const int grid_size = 24;
    const int kernel_size = 3;  // 3×3邻域
    const float coverage_thresh = 0.9f;  // 单个点被视为“已覆盖”的阈值

    // 初始化掩码：[batch, 576]（24×24的1D展开）
    auto mask = torch::zeros({ batch_size, grid_size * grid_size }, coverage_map.options());

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < grid_size; ++i) {  // 遍历行
            for (int j = 0; j < grid_size; ++j) {  // 遍历列
                int idx = i * grid_size + j;  // 1D索引
                int total = 0;  // 邻域内有效点总数（在网格内的点）
                int covered = 0;  // 邻域内覆盖度达标的点数量

                // 检查3×3邻域内的所有点
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        int ni = i + di;  // 邻域点行坐标
                        int nj = j + dj;  // 邻域点列坐标

                        // 只统计网格内的点（忽略超出边界的点）
                        if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size) {
                            total++;  // 有效点计数+1
                            // 如果该点覆盖度≥阈值，视为“已覆盖”
                            if (coverage_map[b][ni][nj].item<float>() >= coverage_thresh) {
                                covered++;
                            }
                        }
                    }
                }

                // 计算覆盖比例（覆盖度 = 达标的点 / 总有效点）
                // 确保total≠0（边界点的邻域有效点可能少于9，但至少为1）
                float coverage_ratio = static_cast<float>(covered) / total;
                mask[b][idx] = coverage_ratio;  // 赋值0~1的连续值
            }
        }
    }

    return mask;
}

torch::Tensor LSTMPathDecoder::idx_to_coordinate(const torch::Tensor& indices, int grid_size ) {
    // indices: [batch] 包含0-575的索引
    auto batch_size = indices.size(0);
    auto coords = torch::zeros({ batch_size, 2}, indices.device());
    //std::cout <<"indices:" << indices.sizes() << std::endl;
    for (int b = 0; b < batch_size; ++b) {
        int idx = indices[b].item<int>();
        int y = idx / grid_size;
        int x = idx % grid_size;
        coords[b][0] = static_cast<float>(x);
        coords[b][1] = static_cast<float>(y);
    }
   // std::cout << coords.sizes() << std::endl;
    return coords;//找到索引并返回,返回 [batch_size, 2]
}
torch::Tensor LSTMPathDecoder::update_coverage(const torch::Tensor& coverage_map,
    const torch::Tensor& point_coords) {
    // coverage_map: [batch, 24, 24]
    // point_coords: [batch, 2] (x, y坐标)
    //std::cout << point_coords.sizes() << std::endl;
    //std::cout << coverage_map.sizes() << std::endl;
    auto new_coverage = coverage_map.clone();
    auto batch_size = coverage_map.size(0);

    for (int b = 0; b < batch_size; ++b) {
        int x = static_cast<int>(point_coords[b][0].item<float>());
        int y = static_cast<int>(point_coords[b][1].item<float>());

        // 更新3x3区域
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < 24 && ny >= 0 && ny < 24) {
                    new_coverage[b][ny][nx] = 1.0f;  // 标记为已覆盖
                }
            }
        }
    }

    return new_coverage;
}
int LSTMPathDecoder::get_hidden_size() {
    return hidden_size;
}


//class PathPlanningTrainer
PathPlanningTrainer::PathPlanningTrainer(torch::Device device) : device(device) 
 {
    
    // 初始化模型
    encoder = CNNEncoder();
    decoder = LSTMPathDecoder();

    // 将模型移动到设备
    encoder.to(device);
    decoder.to(device);

    // 正确初始化优化器 - 修正1：使用正确的参数收集方式
    std::vector<torch::Tensor> all_parameters;

    // 收集编码器参数
    for (auto& param : encoder.parameters()) {
        all_parameters.push_back(param);
    }

    // 收集解码器参数  
    for (auto& param : decoder.parameters()) {
        all_parameters.push_back(param);
    }
    
    optimizer = std::make_unique<torch::optim::Adam>(all_parameters, torch::optim::AdamOptions(1e-4));
}
torch::Tensor PathPlanningTrainer::train_step(std::vector<torch::Tensor> region_masks, std::vector<torch::Tensor> vector_fields 
    ,int samples_num , int width , int height ) {
    batch_size = samples_num;

    TORCH_CHECK(region_masks.size() == batch_size, "Batch size mismatch");//安全检查
    
    //1.数据处理
    MultiObjectiveLoss loss_calculator(device, lambda_coverage, lambda_overlap, lambda_smoothness, lambda_alignment);
    //ImprovedSampler sampler(device, 1.0f, 0.1f, 0.9995f);
    InputProcessor processor;
    std::vector<torch::Tensor>samples;
    std::vector<torch::Tensor>upsampled_vector_fields_V;
    for (auto& mask : region_masks) {
        mask = mask.to(device);  // 移到与 states 相同的设备
    }
    for (auto& field : vector_fields) {
        field = field.to(device);  // 移到与 states 相同的设备
    }
    for (int i = 0 ; i < samples_num; i++) {
        torch::Tensor test = processor.process(region_masks[i], vector_fields[i]);
        samples.push_back(test);
        // 上采样矢量场到24×24
        auto upsampled_field = processor.upsample_vector_field_only(vector_fields[i]);
        upsampled_vector_fields_V.push_back(upsampled_field);
    }

    torch::Tensor batch_features = torch::stack(samples, 0);//batch_size,24,24,5

    // 设置为训练模式
    encoder.train();//cnn
    decoder.train();//lstm
    optimizer->zero_grad();//learning rate

   
    batch_features = batch_features.to(device);

    // 2. CNN特征提取
    auto cnn_features = encoder.forward(batch_features);  // [batch_size, 36, 256]

    // 3. 为批量中的每个样本初始化状态
    auto states = LSTMPathDecoder::initialize_batch_states(region_masks, device, decoder.get_hidden_size(),decoder.get_m_layer_num());//每个样本初始化
    std::vector<std::vector<torch::Tensor>> batch_paths(batch_size);
    std::vector<std::vector<torch::Tensor>> batch_coverages(batch_size);
    

    // 4. 批量序列生成
    auto vector_fields_tensor_for_loss = torch::stack(upsampled_vector_fields_V).to(device);
    auto vector_fields_tensor = torch::stack(vector_fields).to(device);

    torch::Tensor temp_loss = torch::zeros({ 1 }, device);//定义临时损失测试梯度消失原因

    for (int step = 0; step < max_path_length; ++step) {
        bool all_terminated = true;
        if (step % 50 == 0) {
            std::cout << "step:" << step+1 << std::endl;
        }
        // 处理批量中每个样本的当前步骤
        for (int i = 0; i < batch_size; ++i) {
            if (states[i].step_count >= max_path_length ||
                is_coverage_complete(states[i].coverage_map, region_masks[i])) {
                continue;  // 跳过已终止的样本
            }

            all_terminated = false;

            // 获取当前样本的CNN特征
            auto sample_cnn_features = cnn_features[i].unsqueeze(0);  // [1, 36, 256]
            auto sample_vector_field = vector_fields_tensor[i].unsqueeze(0);  // [1, 24, 24, 2]
            //std::cout << sample_cnn_features.sizes() << std::endl;
            //std::cout << sample_vector_field.sizes() << std::endl;

            // 解码器前向传播
            auto [point_logits, new_state] = decoder.forward(
                sample_cnn_features, states[i], sample_vector_field);

            // 采样下一个点
             // 方案：温度退火Gumbel-Softmax
            // 使用可微分采样
            auto [straight_through_samples, hard_samples] =
                sampler.sample_with_temperature_st(point_logits);

            auto next_point_idx = hard_samples;//硬采样
            auto next_point_batch = LSTMPathDecoder::idx_to_coordinate(next_point_idx);//硬采样
            auto next_point = next_point_batch.squeeze(0);

            // Straight-Through样本计算损失（保持梯度）
        // straight_through_samples 是 [1, 576] 的概率分布
        // 我们需要将其转换为连续坐标
            auto continuous_coords = sampler.probability_to_continuous_coordinates(straight_through_samples, 24);
            auto continuous_point = continuous_coords.squeeze(0);  // [2]

            //auto straight_through_coords = LSTMPathDecoder::idx_to_coordinate(straight_through_samples);
            //auto straight_through_point = straight_through_coords.squeeze(0);
            // 
            // 
           // auto next_point_idx = sampler.sample_with_temperature(point_logits);

            //std::cout << "next_point_idx_size:" << next_point_idx.sizes() << std::endl;
            //auto next_point_batch = LSTMPathDecoder::idx_to_coordinate(next_point_idx);
            //std::cout <<"next_point_batch.sizes:" << next_point_batch.sizes() << std::endl;
            //auto next_point = next_point_batch.squeeze(0);
           // std::cout <<"next_point.sizes:" << next_point.sizes() << std::endl;

            // 更新覆盖状态
            new_state.coverage_map = LSTMPathDecoder::update_coverage(states[i].coverage_map, next_point_batch);
           
            // std::cout << "new_state.coverage_map:" << new_state.coverage_map.sizes() << std::endl;
            //std::cout << "next_point_batch.device():" << next_point_batch.device() << std::endl;
            new_state.last_point = next_point_batch;

            //测试用的
            temp_loss += 1e-3 * torch::norm(continuous_point);  // 累加每个样本每个步骤的临时损失


            // 保存路径点
            batch_paths[i].push_back(continuous_point);
            batch_coverages[i].push_back(new_state.coverage_map);

            states[i] = new_state;
        }

        if (all_terminated) { 
            std::cout << "step:" << step << std::endl;
            break; }
    }

    // 5. 计算批量损失
    auto total_loss = loss_calculator.compute_batch_loss(batch_paths, batch_coverages,
        vector_fields_tensor_for_loss, region_masks)/* + temp_loss / batch_size*/;//加入了临时损失

    // 6. 反向传播
    total_loss.backward();
    std::vector<torch::Tensor> parameters = get_all_parameters();
    print_gradient_info(parameters);
    torch::nn::utils::clip_grad_norm_(parameters, 1.0);
    //print_gradient_info(parameters);
    optimizer->step();

    return total_loss;
}
bool PathPlanningTrainer::is_coverage_complete(const torch::Tensor& coverage_map,
    const torch::Tensor& region_mask) {
    auto highres_mask = input_processor.upsample_3x3(region_mask).squeeze(-1);
    //std::cout <<"highres_mask:" << highres_mask.sizes() << std::endl;
    auto target_area = torch::sum(highres_mask);
    //std::cout << " coverage_map:" << coverage_map.sizes() << std::endl;
    auto coverage_map_one = coverage_map.squeeze(0);
    //std::cout<<" coverage_map_one:" << coverage_map_one.sizes() << std::endl;
    auto covered_area = torch::sum(coverage_map_one * highres_mask);
    float coverage_ratio = covered_area.item<float>() / target_area.item<float>();
    return coverage_ratio >= 0.95f;  // 95%覆盖视为完成
}
torch::Tensor PathPlanningTrainer::update_coverage(const torch::Tensor& coverage_map,
    const torch::Tensor& point_coords) {
    auto new_coverage = coverage_map.clone();
    auto batch_size = coverage_map.size(0);

    for (int b = 0; b < batch_size; ++b) {
        int x = static_cast<int>(point_coords[b][0].item<float>());
        int y = static_cast<int>(point_coords[b][1].item<float>());

        // 更新3x3区域
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < coverage_map.size(2) &&
                    ny >= 0 && ny < coverage_map.size(1)) {
                    new_coverage[b][ny][nx] = 1.0f;
                }
            }
        }
    }

    return new_coverage;
}
std::vector<torch::Tensor> PathPlanningTrainer::get_all_parameters() {
    std::vector<torch::Tensor> all_params;

    // 获取编码器参数
    auto encoder_params = encoder.parameters();
    for (auto& param : encoder_params) {
        all_params.push_back(param);
    }

    // 获取解码器参数
    auto decoder_params = decoder.parameters();
    for (auto& param : decoder_params) {
        all_params.push_back(param);
    }

    return all_params;
}

void PathPlanningTrainer::print_gradient_info(const std::vector<torch::Tensor>& parameters) {
    std::cout << "=== Gradient Information ===" << std::endl;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param = parameters[i];
        if (param.grad().defined()) {
            auto grad_norm = torch::norm(param.grad());
            std::cout << "Parameter " << i << ":\n";
            std::cout << "  Shape: " << param.sizes() << "\n";
            std::cout << "  Gradient Norm: " << grad_norm.item<float>() << "\n";
            std::cout << "  Gradient Min: " << param.grad().min().item<float>() << "\n";
            std::cout << "  Gradient Max: " << param.grad().max().item<float>() << "\n";
        }
        else {
            std::cout << "Parameter " << i << " has no gradient\n";
        }
    }
}

//class MultiObjectiveLoss
torch::Tensor MultiObjectiveLoss::compute_batch_loss(
    const std::vector<std::vector<torch::Tensor>>& batch_paths,      // [batch_size][path_length][batch_size, 2]
    const std::vector<std::vector<torch::Tensor>>& batch_coverages,  // [batch_size][path_length, 24, 24]
    const torch::Tensor& vector_fields,                              // [batch_size, 24, 24, 2]
    const std::vector<torch::Tensor>& region_masks) {                // [batch_size, 8, 8]

    int batch_size = batch_paths.size();
    torch::Tensor total_loss = torch::zeros({ 1 }, device);
    int num_valid_samples = 0;

    for (int i = 0; i < batch_size; ++i) {
        if (batch_paths[i].empty()) continue;

        auto sample_loss = compute_sample_loss(
            batch_paths[i],
            batch_coverages[i],
            vector_fields[i],
            region_masks[i]
        );

        total_loss += sample_loss;
        num_valid_samples++;
    }

    if (num_valid_samples > 0) {
        total_loss = total_loss / num_valid_samples;
    }

    return total_loss;
}
torch::Tensor MultiObjectiveLoss::compute_sample_loss(
    const std::vector<torch::Tensor>& path_points,      // [path_length],[batch_size, 2]
    const std::vector<torch::Tensor>& coverage_maps,    // [path_length, 24, 24]  
    const torch::Tensor& vector_field,                  // [24, 24, 2]
    const torch::Tensor& region_mask) {                 // [8, 8]

    auto coverage_loss = compute_coverage_loss(coverage_maps.back(), region_mask);
    auto overlap_loss = compute_overlap_loss(path_points, coverage_maps);
    auto smoothness_loss = compute_smoothness_loss(path_points);
    auto alignment_loss = compute_alignment_loss(path_points, vector_field);

    return /*lambda_coverage * coverage_loss +
        lambda_overlap * overlap_loss +
        lambda_smoothness * smoothness_loss +*/
        lambda_alignment * alignment_loss;
}
// 1. 覆盖损失 - 计算未覆盖区域的比例
torch::Tensor MultiObjectiveLoss::compute_coverage_loss(const torch::Tensor& final_coverage,
    const torch::Tensor& region_mask) {
    // 将8×8区域掩码上采样到24×24
    auto highres_mask = upsample_mask_3x3(region_mask).to(device);  // [24, 24]

    // 只计算需要覆盖区域的未覆盖部分
    auto target_area = torch::sum(highres_mask);
    auto covered_area = torch::sum(final_coverage * highres_mask);
    auto uncovered_area = target_area - covered_area;

    // 避免除零
    if (target_area.item<float>() < 1e-6) {
        return torch::zeros({ 1 }, device);
    }

    return uncovered_area / target_area;
}
// 2. 重叠损失 - 计算非相邻点之间的重叠
torch::Tensor MultiObjectiveLoss::compute_overlap_loss(const std::vector<torch::Tensor>& path_points,
    const std::vector<torch::Tensor>& coverage_maps) {
    if (path_points.size() < 3) {
        return torch::zeros({ 1 }, device);
    }

    torch::Tensor total_overlap = torch::zeros({ 1 }, device);
    int overlap_count = 0;

    // 为每个点计算其覆盖区域
    std::vector<torch::Tensor> point_coverages;
    for (const auto& point : path_points) {
        //std::cout << point.sizes() << std::endl;
        //std::cout << point << std::endl;
        auto point_flat = point.flatten();
        TORCH_CHECK(point_flat.size(0) == 2, "Point must have exactly 2 elements");
        point_coverages.push_back(get_point_coverage(point));
    }
    //int num_points = static_cast<int>(path_points.size());

    for (size_t i = 2; i < path_points.size(); ++i) {
        auto current_coverage = point_coverages[i];  // 当前点的覆盖区域

        // 累计之前所有非相邻点的覆盖区域（排除i-1和i-2）
        torch::Tensor previous_coverage = torch::zeros({ 24, 24 }, device);
        for (size_t j = 0; j < i - 2; ++j) {
            previous_coverage += point_coverages[j];
        }
        previous_coverage = torch::clamp(previous_coverage, 0.0f, 1.0f);

        // 计算重叠区域
        auto overlap = current_coverage * previous_coverage;
        total_overlap += torch::sum(overlap);
        overlap_count++;
    }

    if (overlap_count > 0) {
        return total_overlap / overlap_count;
    }
    return torch::zeros({ 1 }, device);
}
// 3. 平滑损失 - 计算路径转折角度
torch::Tensor MultiObjectiveLoss::compute_smoothness_loss(const std::vector<torch::Tensor>& path_points) {
    if (path_points.size() <= 2) {
        return torch::zeros({ 1 }, device);
    }

    torch::Tensor total_angle_loss = torch::zeros({ 1 }, device);
    int angle_count = 0;// 有效夹角计算次数

    for (size_t i = 2; i < path_points.size(); ++i) {
        // 1. 计算相邻两个线段的向量（方向至关重要）
        auto point1 = path_points[i-2].flatten();
        auto point2 = path_points[i - 1].flatten();
        auto point3 = path_points[i].flatten();
        TORCH_CHECK(point1.size(0) == 2 && point2.size(0) == 2 && point3.size(0) == 2,
            "All points must have exactly 2 elements");
        // v1：从第i-2个点指向第i-1个点的向量
        auto v1 = (path_points[i - 1] - path_points[i - 2]).to(device);  // 前一段向量
        // v2：从第i-1个点指向第i个点的向量
        auto v2 = (path_points[i] - path_points[i - 1]).to(device);    // 当前段向量

        // 2. 计算向量的L2范数（线段长度，用于归一化）
        auto norm_v1 = torch::norm(v1);
        auto norm_v2 = torch::norm(v2);

        // 3.避免零向量
        if (norm_v1.item<float>() < 1e-6 || norm_v2.item<float>() < 1e-6) {
            continue;
        }
        // 4. 计算夹角的余弦值（cosθ = (v1·v2) / (||v1||×||v2||)）

        auto cos_angle = torch::dot(v1, v2) / (norm_v1 * norm_v2);
        auto angle_loss = 0.5f * (1.0f - cos_angle);  // 目标：角度接近0°

        total_angle_loss += angle_loss;
        angle_count++;
    }

    if (angle_count > 0) {
        return total_angle_loss / angle_count;
    }
    return torch::zeros({ 1 }, device);
}
// 4. 场对齐损失 - 计算路径段与矢量场的对齐程度
torch::Tensor MultiObjectiveLoss::compute_alignment_loss(const std::vector<torch::Tensor>& path_points,
    const torch::Tensor& vector_field) {
    if (path_points.size() <= 1) {
        return torch::zeros({ 1 }, device);
    }
    //std::cout << vector_field.sizes() << std::endl;
    auto soft_min = [](const torch::Tensor& a, const torch::Tensor& b, float beta = 10.0f) {
        auto exp_neg_a = torch::exp(-beta * a);
        auto exp_neg_b = torch::exp(-beta * b);
        return (-torch::log(exp_neg_a + exp_neg_b) / beta);
        };


    torch::Tensor total_alignment_loss = torch::zeros({ 1 }, device);
    int segment_count = 0;
    //std::cout << path_points.size() << std::endl;
    auto field_on_device = vector_field.to(device);

    for (size_t i = 1; i < path_points.size(); ++i) {
       // std::cout << i << std::endl;
        auto start_point = path_points[i - 1].to(device);
        auto end_point = path_points[i].to(device);
        TORCH_CHECK(start_point.size(0) == 2 && end_point.size(0) == 2,
            "Start and end points must have exactly 2 elements");

        start_point = start_point.to(device);
        end_point = end_point.to(device);
        auto segment_vector = end_point - start_point;

        // 获取起点处的场方向
        auto field_at_point = bilinear_sample_vector_field(field_on_device, start_point); 
        //std::cout << field_at_point.sizes() << std::endl;
        auto norm_segment = torch::norm(segment_vector);
        auto norm_field = torch::norm(field_at_point);

        // 避免零向量
        if (norm_segment.item<float>() < 1e-6) {
            segment_vector = segment_vector + 1e-6 * torch::randn_like(segment_vector);
            norm_segment = torch::norm(segment_vector);
        }
        if (norm_field.item<float>() < 1e-6) {
            field_at_point = field_at_point + 1e-6 * torch::randn_like(field_at_point);
            norm_field = torch::norm(field_at_point);
        }

        // 计算方向一致性（考虑顺场和逆场都可以）
        //auto dot_product = torch::dot(segment_vector, field_at_point);
        //auto cos_angle = dot_product / (norm_segment * norm_field);
        //auto alignment_loss = 1.0f - torch::abs(cos_angle);  // 目标：角度接近0°或180°
        // 单位化方向向量
        auto segment_dir = segment_vector / norm_segment;
        auto field_dir = field_at_point / norm_field;

        // 计算顺/逆场差异的L2范数
        auto diff_forward_norm = torch::norm(segment_dir - field_dir);  // 顺场差异
        auto diff_backward_norm = torch::norm(segment_dir + field_dir); // 逆场差异

        // 使用soft-min替代硬min，避免梯度截断
        auto min_diff = soft_min(diff_forward_norm, diff_backward_norm, 10.0f);

        // 平方惩罚增强梯度
        auto alignment_loss = min_diff * min_diff;


        total_alignment_loss += alignment_loss;
        segment_count++;
    }

    if (segment_count > 0) {
        return total_alignment_loss / segment_count;
    }
    return torch::zeros({ 1 }, device);
}
// 辅助函数：上采样掩码
torch::Tensor MultiObjectiveLoss::upsample_mask_3x3(const torch::Tensor& mask) {
    int h = mask.size(0);
    int w = mask.size(1);
    auto result = torch::zeros({ h * 3, w * 3 });

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float value = mask[i][j].item<float>();
            result.index_put_(
                { torch::indexing::Slice(i * 3, i * 3 + 3),
                 torch::indexing::Slice(j * 3, j * 3 + 3) },
                value
            );
        }
    }
    return result;
}
// 辅助函数：获取点的覆盖区域
torch::Tensor MultiObjectiveLoss::get_point_coverage(const torch::Tensor& point) {
    torch::Tensor point_flat;
    if (point.dim() == 2 && point.size(0) == 1 && point.size(1) == 2) {
        // 形状为 [1, 2] 的情况
        point_flat = point.squeeze(0);  // 变为 [2]
    }
    else if (point.dim() == 1 && point.size(0) == 2) {
        // 形状为 [2] 的情况
        point_flat = point;
    }
    else {
        TORCH_CHECK(false, "Point must be 1D tensor with 2 elements or 2D tensor with shape [1, 2], got shape: ", point.sizes());
    }
    //int x = static_cast<int>(point[0].item<float>());
    //int y = static_cast<int>(point[1].item<float>());
    int x = static_cast<int>(std::round(point_flat[0].item<float>()));
    int y = static_cast<int>(std::round(point_flat[1].item<float>()));

    auto coverage = torch::zeros({ 24, 24 }, device);
    int start_x = std::max(0, x - 1);
    int end_x = std::min(23, x + 1);
    int start_y = std::max(0, y - 1);
    int end_y = std::min(23, y + 1);

    coverage.index_put_(
        { torch::indexing::Slice(start_y, end_y + 1),
         torch::indexing::Slice(start_x, end_x + 1) },
        1.0f
    );

    return coverage;
}

// 辅助函数：双线性采样矢量场
torch::Tensor MultiObjectiveLoss::bilinear_sample_vector_field(const torch::Tensor& vector_field,
    const torch::Tensor& point) {
    // 确保point是正确形状的
    TORCH_CHECK(point.dim() == 1 && point.size(0) == 2,
        "Point must be 1D tensor with 2 elements for sampling");
    TORCH_CHECK(point.requires_grad(), "point 未跟踪梯度，无法传递梯度！");

    // vector_field: [1, 2, 24, 24]（batch=1, channels=2(x/y分量), height=24, width=24）
   // 需先将 [24,24,2] 转为 PyTorch 采样要求的 [N, C, H, W] 格式
    auto field = vector_field.permute({ 2, 0, 1 }).unsqueeze(0);  // [1, 2, 24, 24]

    // grid_sample 要求坐标范围：x/y ∈ [-1, 1]（与你数据集的归一化范围一致）
    // point: [2] → 转为 [1, 1, 1, 2]（N, H, W, 2），表示“1个采样点”
    auto grid = point.unsqueeze(0).unsqueeze(0).unsqueeze(0);  // [1,1,1,2]
    grid = (grid / 23.0f) * 2.0f - 1.0f;  // [0,23] → [-1,1]
    // 3. 可导双线性采样
    auto sampled = torch::nn::functional::grid_sample(
        field,
        grid,
        torch::nn::functional::GridSampleFuncOptions()
        .mode(torch::kBilinear)  // 双线性插值
        .padding_mode(torch::kBorder)  // 边界用原数值填充
        .align_corners(true)  // 确保角落坐标采样正确
    );
    auto result = sampled.squeeze();
  

    return result;
}

//class ImprovedSampler
 // 方案：返回硬采样和连续坐标
std::pair<torch::Tensor, torch::Tensor> ImprovedSampler::sample_with_temperature_st(const torch::Tensor& point_logits) {
    float current_temp = get_current_temperature();

    // 确保logits不会太大导致softmax饱和
    auto stabilized_logits = point_logits - torch::max(point_logits);

    // 应用温度缩放
    auto tempered_logits = stabilized_logits / current_temp;
    // 添加适度的Gumbel噪声
    auto gumbel_noise = -torch::log(-torch::log(torch::rand_like(tempered_logits) + 1e-8) + 1e-8);
    auto noisy_logits = tempered_logits + gumbel_noise;

    auto soft_samples = torch::softmax(noisy_logits, -1);
    auto hard_samples = torch::argmax(soft_samples, -1);
    
    // 添加Gumbel噪声
   
    //auto gumbel_noise = -torch::log(-torch::log(torch::rand_like(point_logits)));

    // 应用温度缩放
    //auto tempered_logits = (point_logits + gumbel_noise) / current_temp;
    //auto soft_samples = torch::softmax(tempered_logits, -1);

    // 硬采样：在前向传播中使用argmax
    //auto hard_samples = torch::argmax(soft_samples, -1);

    // Straight-Through技巧：在前向传播中使用硬采样，在反向传播中使用软采样
    auto straight_through_samples = hard_samples + soft_samples - soft_samples.detach();
    //straight_through_samples = straight_through_samples.squeeze(0);
    //std::cout<<"straight_through_samples:" << straight_through_samples.sizes() << std::endl;
    //std::cout <<"hard_samples:" << hard_samples.sizes() << std::endl;
    training_step++;
    return { straight_through_samples, hard_samples };
}
torch::Tensor ImprovedSampler::sample_with_temperature(const torch::Tensor& point_logits) {
    auto [hard_samples, _] = sample_with_temperature_st(point_logits);
    return hard_samples;
}
// 将概率分布转换为连续坐标
torch::Tensor ImprovedSampler::probability_to_continuous_coordinates(const torch::Tensor& probs, int grid_size) {
    auto batch_size = probs.size(0);
    auto coords = torch::zeros({ batch_size, 2 }, probs.device());

    // 预计算所有可能的坐标
    auto grid_coords = torch::zeros({ grid_size * grid_size, 2 }, probs.device());
    for (int i = 0; i < grid_size * grid_size; ++i) {
        grid_coords[i][0] = static_cast<float>(i % grid_size);
        grid_coords[i][1] = static_cast<float>(i / grid_size);
    }

    // 计算期望坐标：probs [batch, 576] × grid_coords [576, 2] = [batch, 2]
    coords = torch::matmul(probs, grid_coords);
    //std::cout <<"coords:" << coords.sizes() << std::endl;
    return coords;
}

