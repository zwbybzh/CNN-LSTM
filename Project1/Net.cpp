#include "Net.hpp"
#include "FieldDataset.hpp"
//class InputProcessor
torch::Tensor InputProcessor::process(const torch::Tensor& region_mask,
    const torch::Tensor& vector_field) {//ע��������״�����볡��״Ҫһ��
    torch::Device device = region_mask.device();  // ��ȡ�����豸����GPU��
    // 1. ���������ϲ�����24x24 (�Ź���ϸ��)
    auto highres_mask = upsample_3x3(region_mask);

    // 2. ʸ�����ϲ�����24x24x2
    auto highres_field = upsample_vector_field(vector_field);

    int64_t original_height = vector_field.size(0);
    int64_t original_width = vector_field.size(1);
    // 3. ����λ�ñ�������
    auto grid = create_position_grid(original_height*3, original_width*3).to(device);//Ӧ���봫�����������һ�£�

    // 4. ƴ����������: [����, ������, λ��] �� 24x24x5
    return torch::cat({ highres_mask, highres_field, grid }, -1);
}
torch::Tensor InputProcessor::upsample_3x3(const torch::Tensor& mask) {
    // ����: 8x8, ���: 24x24
    TORCH_CHECK(mask.dim() == 2, "�������������2D������ʵ��ά��: ", mask.dim());
    // ������mask��ȡԭʼά�� [H, W]
    int64_t original_height = mask.size(0);
    int64_t original_width = mask.size(1);//8x8Ϊ��
    //std::cout << original_height << original_width << std::endl;
    // ����3���ϲ������Ŀ��ά��
    int64_t target_height = original_height * 3;
    int64_t target_width = original_width * 3;

    auto options = torch::TensorOptions().dtype(mask.dtype()).device(mask.device());
    auto result = torch::zeros({ target_height, target_width }, options);

    for (int i = 0; i < original_height; ++i) {
        for (int j = 0; j < original_width; ++j) {
            float value = mask[i][j].item<float>();
            // ��ÿ��8x8��������չΪ3x3��
            result.index_put_({ torch::indexing::Slice(i * 3, i * 3 + 3),
                             torch::indexing::Slice(j * 3, j * 3 + 3) }, value);
        }
    }
    return result.unsqueeze(-1);  // 24x24x1
}
torch::Tensor InputProcessor::upsample_vector_field(const torch::Tensor& vector_field) {
    // ����: 8x8x2, ���: 24x24x2
    TORCH_CHECK(vector_field.dim() == 3,
        "ʸ����������3D���� [H, W, 2]��ʵ��ά��: ", vector_field.dim());
    TORCH_CHECK(vector_field.size(2) == 2,
        "ʸ�������һά����Ϊ2��x��y��������ʵ�ʳߴ�: ", vector_field.size(2));
    

    // ����1: �򵥵��ظ���ֵ
    auto field_x = vector_field.index({ "...", 0 });  // 8x8
    auto field_y = vector_field.index({ "...", 1 });  // 8x8

    // �ϲ���x����
    auto upsampled_x = upsample_3x3_smooth(field_x);
    auto upsampled_y = upsample_3x3_smooth(field_y);

    // �ϲ�Ϊ24x24x2
    return torch::stack({ upsampled_x, upsampled_y }, -1);
}
torch::Tensor InputProcessor::upsample_3x3_smooth(const torch::Tensor& field) {

    TORCH_CHECK(field.dim() == 2, "���������2D������ʵ��ά��: ", field.dim());

    // �����붯̬��ȡԭʼά��
    int64_t original_height = field.size(0);
    int64_t original_width = field.size(1);

    // ����3���ϲ������Ŀ��ά��
    int64_t target_height = original_height * 3;
    int64_t target_width = original_width * 3;

    // ����������ͬ���͡�ͬ�豸���������
    auto options = torch::TensorOptions().dtype(field.dtype()).device(field.device());
    auto result = torch::zeros({ target_height, target_width }, options);

    for (int i = 0; i < original_height; ++i) {
        for (int j = 0; j < original_width; ++j) {
            float value = field[i][j].item<float>();

            // ��ȡ����ֵ���ڲ�ֵ
            float left = (j > 0) ? field[i][j - 1].item<float>() : value;
            float right = (j < original_width - 1) ? field[i][j + 1].item<float>() : value;
            float top = (i > 0) ? field[i - 1][j].item<float>() : value;
            float bottom = (i < original_height - 1) ? field[i + 1][j].item<float>() : value;

            // 3x3���ڵĲ�ֵȨ��
            for (int di = 0; di < 3; ++di) {
                for (int dj = 0; dj < 3; ++dj) {
                    float weight_x = 1.0f - std::abs(dj - 1.0f) / 1.0f;  // ˮƽȨ��
                    float weight_y = 1.0f - std::abs(di - 1.0f) / 1.0f;  // ��ֱȨ��

                    float interpolated = value;
                    if (dj == 0 && j > 0) {  // ��߽磬�����ֵ
                        interpolated = value * 0.7f + left * 0.3f;
                    }
                    else if (dj == 2 && j < original_width - 1) {  // �ұ߽磬���Ҳ�ֵ
                        interpolated = value * 0.7f + right * 0.3f;
                    }
                    else if (di == 0 && i > 0) {  // �ϱ߽磬���ϲ�ֵ
                        interpolated = value * 0.7f + top * 0.3f;
                    }
                    else if (di == 2 && i < original_height - 1) {  // �±߽磬���²�ֵ
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

    // ������һ������ [-1, 1]
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float norm_x = (2.0f * j / (width - 1)) - 1.0f;  // x��-1��1
            float norm_y = (2.0f * i / (height - 1)) - 1.0f; // y��-1��1

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
    // ��һ��: ��ȡ�ֲ�����
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(5, 64, 3).stride(1).padding(1)));  // ����5ͨ��, ���64ͨ��
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

    // �ڶ���: �еȳ߶�����
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));

    // ������: ȫ������
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
    // ����ʽ�� [batch, height, width, channels] ת��Ϊ [batch, channels, height, width]
    auto x_permuted = x.permute({ 0, 3, 1, 2 });  // ������״�� [batch, 5, 24, 24]

    // ��һ����: [batch, 5, 24, 24] �� [batch, 64, 24, 24]
    auto h = torch::relu(bn1(conv1(x_permuted)));

    // ��һ�γػ�: [batch, 64, 24, 24] �� [batch, 128, 12, 12]
    h = pool(torch::relu(bn2(conv2(h))));

    // �ڶ��γػ�: [batch, 128, 12, 12] �� [batch, 256, 6, 6]
    h = pool(torch::relu(bn3(conv3(h))));

    // չƽ����ͼ: [batch, 256, 6, 6] �� [batch, 36, 256]
    // ���ռ�ά��չƽΪ���У�����ͨ��ά����Ϊ����
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
//    // �������
//    auto batch_size = h3_pooled.size(0);
//    auto output = h3_pooled.permute({ 0, 2, 3, 1 }).reshape({ batch_size, 36, 256 });
//    features.push_back(output);
//
//    return features;
//}

//class AttentionLayer
AttentionLayer::AttentionLayer(int hidden_size): hidden_size(hidden_size) {//����ע�ᵽmodule��
        query_proj = register_module("query_proj",
            torch::nn::Linear(hidden_size, hidden_size));//����ά��=���ά��
        key_proj = register_module("key_proj",
            torch::nn::Linear(256, hidden_size));  // CNN����ά��256,���Ϊhidden_sizeһ���Ǽ�ֵ
        value_proj = register_module("value_proj",
            torch::nn::Linear(256, hidden_size));//һ����ֵ
        //��ʼ��

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

    auto query = torch::leaky_relu(query_proj(lstm_hidden),0.1).unsqueeze(1);  // [batch, 1, hidden]ֻ��һ����ѯ����lstm����
    auto keys = torch::leaky_relu(key_proj(cnn_features),0.1);                 // [batch, 36, hidden]
    auto values = torch::leaky_relu(value_proj(cnn_features),0.1);             // [batch, 36, hidden]

    // ����ע����Ȩ��
    auto scores = torch::bmm(query, keys.transpose(1, 2));  // [batch, 1, 36]
    auto attn_weights = torch::softmax(scores, 2);

    // ��Ȩ���
    // ��Ȩ��ͣ�[batch, 1, 36] �� [batch, 36, hidden] �� [batch, 1, hidden] �� [batch, hidden]
    auto context = torch::bmm(attn_weights, values).squeeze(1);  // [batch, hidden]
    return context;
}

//class LSTMPathDecoder
LSTMPathDecoder::LSTMPathDecoder(int hidden_size,int layer_num) : hidden_size(hidden_size) {
    // ��ȷ������ߴ���㣺
    // point_embedding���: 32ά
    // coverage_embedding���: 128ά  
    // attention������: hidden_sizeά
    // ������ߴ�: 32 + 128 + hidden_size
    int input_size = 32 + 128 + hidden_size;
    m_layer_num = layer_num;
    // 2��LSTM��batch_first=true
    lstm = register_module("lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
            .num_layers(m_layer_num)
            .batch_first(true)
            /*.dropout(0.1)*/));

    // ʹ��make_shared��ȷ��ʼ��AttentionLayer
    attention = std::make_shared<AttentionLayer>(hidden_size);
    register_module("attention", attention);

    point_embedding = register_module("point_embedding",
        torch::nn::Linear(2, 32));  // 2D����Ƕ�뵽32ά

    coverage_embedding = register_module("coverage_embedding",
        torch::nn::Linear(576, 128));  // ����״̬Ƕ��(24x24=576)

    output_layer = register_module("output_layer",
        torch::nn::Linear(hidden_size, 576));  // ���576����ĸ���

    // ��ѡ�ģ�����ͶӰ�㣬ȷ��CNN����ά��ƥ��
    /*feature_projection = register_module("feature_projection",
        torch::nn::Linear(256, 256));*/
    //��ʼ��Ȩ��
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
    DecoderState state,                  // ������״̬
    const torch::Tensor& vector_field    // [batch, 24, 24, 2] ʸ����,����û����
) {
    auto batch_size = cnn_features.size(0);
   // std::cout <<"LSTMPathDecoder_batch_size:" << batch_size << std::endl;
    // 1. ׼��LSTM��������

    // ������Ƕ��: [batch, 2] -> [batch, 32]
    //std::cout << "state.last_point" << std::endl;
   // std::cout << state.last_point << std::endl;
    //std::cout << state.last_point.device() << std::endl;
    auto point_embed = torch::leaky_relu(point_embedding(state.last_point), 0.1);
   // std::cout <<"point_embed:" << point_embed.sizes() << std::endl;
    // ����״̬Ƕ��: [batch, 24, 24] -> [batch, 576] -> [batch, 128]
    auto coverage_flat = state.coverage_map.reshape({ batch_size, -1 });  // [batch, 576]
    auto coverage_embed = torch::leaky_relu(coverage_embedding(coverage_flat), 0.1); // [batch, 128]
    //std::cout<<"coverage_embed.sizes():" << coverage_embed.sizes() << std::endl;
    //std::cout << state.hidden_state[-1].sizes() << std::endl;

    // 2. ����ע����������
    // ʹ��LSTM���һ������һ��ʱ�䲽������״̬
    auto last_hidden = state.hidden_state[-1];  // [batch, hidden_size]
    
    auto context = attention->forward(last_hidden, cnn_features);  // [batch, hidden_size]

    // 3. ƴ����������: [batch, 32+128+hidden_size]
    auto lstm_input = torch::cat({ point_embed, coverage_embed, context }, 1);
    lstm_input = lstm_input.unsqueeze(1);  // [batch, 1, input_size]

    // 4. LSTMǰ�򴫲�
    auto lstm_output = lstm->forward(lstm_input,
        std::pair<torch::Tensor, torch::Tensor>{state.hidden_state, state.cell_state });

    auto lstm_out = std::get<0>(lstm_output);  // [batch, 1, hidden_size]
    auto new_states = std::get<1>(lstm_output);

    // 5. �����: Ԥ����һ����ĸ��ʷֲ�
    auto lstm_hidden = lstm_out.squeeze(1);  // [batch, hidden_size]
    auto point_logits = output_layer(lstm_hidden);  // [batch, 576]

    // 6.1 Ӧ�ø�������
    auto coverage_mask = create_coverage_mask(state.coverage_map);  // [batch, 576]
    point_logits = point_logits - 1* coverage_mask;// �Ѹ�������ĵ㼸�����ᱻѡ��(�ͷ������е�࣡��������)
    // 6.2���Ƶ��õ��7*7����������ʴ�
    auto last_x = state.last_point.index({ torch::indexing::Slice(), 0 }).to(torch::kInt);  // [batch]��x���꣨�У�
    auto last_y = state.last_point.index({ torch::indexing::Slice(), 1 }).to(torch::kInt);  // [batch]��y���꣨�У�
    auto range_mask = torch::ones({ batch_size, 576 }, point_logits.device());
    for (int b = 0; b < batch_size; ++b) {
        int x = last_x[b].item<int>();  // ��ǰ������һ���x����
        int y = last_y[b].item<int>();  // ��ǰ������һ���y����

        // ����7��7����ı߽磨ȷ����0~23��Χ�ڣ�
        int x_min = std::max(0, x - 3);    // x������߽磨��С0��
        int x_max = std::min(23, x + 3);   // x�����ұ߽磨���23��
        int y_min = std::max(0, y - 3);    // y�����ϱ߽磨��С0��
        int y_max = std::min(23, y + 3);   // y�����±߽磨���23��

        // ���7��7�����ڵĵ㣺������Ϊ0�����ͷ���
        for (int yi = y_min; yi <= y_max; ++yi) {
            for (int xi = x_min; xi <= x_max; ++xi) {
                int idx = yi * 24 + xi;  // ���������0~575��
                range_mask[b][idx] = 0.0f;  // 7��7�����ڵĵ㲻�ͷ�
            }
        }
    }
    point_logits = point_logits - 1 * range_mask;//������Ϊͬ���ĸ���

    // 7. ���½�����״̬
    torch::Tensor sam = state.coverage_map[0];//24x24
    DecoderState new_state(sam, batch_size);
    new_state.hidden_state = std::get<0>(new_states);
    new_state.cell_state = std::get<1>(new_states);

    new_state.coverage_map = state.coverage_map;  // ���ⲿ����
    new_state.last_point = state.last_point;      // ���ⲿ���� 

    new_state.step_count = state.step_count + 1;

    return { point_logits, new_state };
}
torch::Tensor LSTMPathDecoder::create_coverage_mask(const torch::Tensor& coverage_map) {
    //// coverage_map: [batch, 24, 24]
    //auto batch_size = coverage_map.size(0);

    //// �������룺������Ӧ��3x3��������ȫ���ǣ�������Ϊ1
    //auto mask = torch::zeros({ batch_size, 576 });//����Ҳ��Ҫ��

    //for (int b = 0; b < batch_size; ++b) {
    //    for (int i = 0; i < 24; ++i) {
    //        for (int j = 0; j < 24; ++j) {
    //            int idx = i * 24 + j;

    //            // ���3x3�����Ƿ�����ȫ����
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
    //                        fully_covered = false;  // �߽���������ȫ����
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
    const int kernel_size = 3;  // 3��3����
    const float coverage_thresh = 0.9f;  // �����㱻��Ϊ���Ѹ��ǡ�����ֵ

    // ��ʼ�����룺[batch, 576]��24��24��1Dչ����
    auto mask = torch::zeros({ batch_size, grid_size * grid_size }, coverage_map.options());

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < grid_size; ++i) {  // ������
            for (int j = 0; j < grid_size; ++j) {  // ������
                int idx = i * grid_size + j;  // 1D����
                int total = 0;  // ��������Ч���������������ڵĵ㣩
                int covered = 0;  // �����ڸ��Ƕȴ��ĵ�����

                // ���3��3�����ڵ����е�
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        int ni = i + di;  // �����������
                        int nj = j + dj;  // �����������

                        // ֻͳ�������ڵĵ㣨���Գ����߽�ĵ㣩
                        if (ni >= 0 && ni < grid_size && nj >= 0 && nj < grid_size) {
                            total++;  // ��Ч�����+1
                            // ����õ㸲�Ƕȡ���ֵ����Ϊ���Ѹ��ǡ�
                            if (coverage_map[b][ni][nj].item<float>() >= coverage_thresh) {
                                covered++;
                            }
                        }
                    }
                }

                // ���㸲�Ǳ��������Ƕ� = ���ĵ� / ����Ч�㣩
                // ȷ��total��0���߽���������Ч���������9��������Ϊ1��
                float coverage_ratio = static_cast<float>(covered) / total;
                mask[b][idx] = coverage_ratio;  // ��ֵ0~1������ֵ
            }
        }
    }

    return mask;
}

torch::Tensor LSTMPathDecoder::idx_to_coordinate(const torch::Tensor& indices, int grid_size ) {
    // indices: [batch] ����0-575������
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
    return coords;//�ҵ�����������,���� [batch_size, 2]
}
torch::Tensor LSTMPathDecoder::update_coverage(const torch::Tensor& coverage_map,
    const torch::Tensor& point_coords) {
    // coverage_map: [batch, 24, 24]
    // point_coords: [batch, 2] (x, y����)
    //std::cout << point_coords.sizes() << std::endl;
    //std::cout << coverage_map.sizes() << std::endl;
    auto new_coverage = coverage_map.clone();
    auto batch_size = coverage_map.size(0);

    for (int b = 0; b < batch_size; ++b) {
        int x = static_cast<int>(point_coords[b][0].item<float>());
        int y = static_cast<int>(point_coords[b][1].item<float>());

        // ����3x3����
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < 24 && ny >= 0 && ny < 24) {
                    new_coverage[b][ny][nx] = 1.0f;  // ���Ϊ�Ѹ���
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
    
    // ��ʼ��ģ��
    encoder = CNNEncoder();
    decoder = LSTMPathDecoder();

    // ��ģ���ƶ����豸
    encoder.to(device);
    decoder.to(device);

    // ��ȷ��ʼ���Ż��� - ����1��ʹ����ȷ�Ĳ����ռ���ʽ
    std::vector<torch::Tensor> all_parameters;

    // �ռ�����������
    for (auto& param : encoder.parameters()) {
        all_parameters.push_back(param);
    }

    // �ռ�����������  
    for (auto& param : decoder.parameters()) {
        all_parameters.push_back(param);
    }
    
    optimizer = std::make_unique<torch::optim::Adam>(all_parameters, torch::optim::AdamOptions(1e-4));
}
torch::Tensor PathPlanningTrainer::train_step(std::vector<torch::Tensor> region_masks, std::vector<torch::Tensor> vector_fields 
    ,int samples_num , int width , int height ) {
    batch_size = samples_num;

    TORCH_CHECK(region_masks.size() == batch_size, "Batch size mismatch");//��ȫ���
    
    //1.���ݴ���
    MultiObjectiveLoss loss_calculator(device, lambda_coverage, lambda_overlap, lambda_smoothness, lambda_alignment);
    //ImprovedSampler sampler(device, 1.0f, 0.1f, 0.9995f);
    InputProcessor processor;
    std::vector<torch::Tensor>samples;
    std::vector<torch::Tensor>upsampled_vector_fields_V;
    for (auto& mask : region_masks) {
        mask = mask.to(device);  // �Ƶ��� states ��ͬ���豸
    }
    for (auto& field : vector_fields) {
        field = field.to(device);  // �Ƶ��� states ��ͬ���豸
    }
    for (int i = 0 ; i < samples_num; i++) {
        torch::Tensor test = processor.process(region_masks[i], vector_fields[i]);
        samples.push_back(test);
        // �ϲ���ʸ������24��24
        auto upsampled_field = processor.upsample_vector_field_only(vector_fields[i]);
        upsampled_vector_fields_V.push_back(upsampled_field);
    }

    torch::Tensor batch_features = torch::stack(samples, 0);//batch_size,24,24,5

    // ����Ϊѵ��ģʽ
    encoder.train();//cnn
    decoder.train();//lstm
    optimizer->zero_grad();//learning rate

   
    batch_features = batch_features.to(device);

    // 2. CNN������ȡ
    auto cnn_features = encoder.forward(batch_features);  // [batch_size, 36, 256]

    // 3. Ϊ�����е�ÿ��������ʼ��״̬
    auto states = LSTMPathDecoder::initialize_batch_states(region_masks, device, decoder.get_hidden_size(),decoder.get_m_layer_num());//ÿ��������ʼ��
    std::vector<std::vector<torch::Tensor>> batch_paths(batch_size);
    std::vector<std::vector<torch::Tensor>> batch_coverages(batch_size);
    

    // 4. ������������
    auto vector_fields_tensor_for_loss = torch::stack(upsampled_vector_fields_V).to(device);
    auto vector_fields_tensor = torch::stack(vector_fields).to(device);

    torch::Tensor temp_loss = torch::zeros({ 1 }, device);//������ʱ��ʧ�����ݶ���ʧԭ��

    for (int step = 0; step < max_path_length; ++step) {
        bool all_terminated = true;
        if (step % 50 == 0) {
            std::cout << "step:" << step+1 << std::endl;
        }
        // ����������ÿ�������ĵ�ǰ����
        for (int i = 0; i < batch_size; ++i) {
            if (states[i].step_count >= max_path_length ||
                is_coverage_complete(states[i].coverage_map, region_masks[i])) {
                continue;  // ��������ֹ������
            }

            all_terminated = false;

            // ��ȡ��ǰ������CNN����
            auto sample_cnn_features = cnn_features[i].unsqueeze(0);  // [1, 36, 256]
            auto sample_vector_field = vector_fields_tensor[i].unsqueeze(0);  // [1, 24, 24, 2]
            //std::cout << sample_cnn_features.sizes() << std::endl;
            //std::cout << sample_vector_field.sizes() << std::endl;

            // ������ǰ�򴫲�
            auto [point_logits, new_state] = decoder.forward(
                sample_cnn_features, states[i], sample_vector_field);

            // ������һ����
             // �������¶��˻�Gumbel-Softmax
            // ʹ�ÿ�΢�ֲ���
            auto [straight_through_samples, hard_samples] =
                sampler.sample_with_temperature_st(point_logits);

            auto next_point_idx = hard_samples;//Ӳ����
            auto next_point_batch = LSTMPathDecoder::idx_to_coordinate(next_point_idx);//Ӳ����
            auto next_point = next_point_batch.squeeze(0);

            // Straight-Through����������ʧ�������ݶȣ�
        // straight_through_samples �� [1, 576] �ĸ��ʷֲ�
        // ������Ҫ����ת��Ϊ��������
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

            // ���¸���״̬
            new_state.coverage_map = LSTMPathDecoder::update_coverage(states[i].coverage_map, next_point_batch);
           
            // std::cout << "new_state.coverage_map:" << new_state.coverage_map.sizes() << std::endl;
            //std::cout << "next_point_batch.device():" << next_point_batch.device() << std::endl;
            new_state.last_point = next_point_batch;

            //�����õ�
            temp_loss += 1e-3 * torch::norm(continuous_point);  // �ۼ�ÿ������ÿ���������ʱ��ʧ


            // ����·����
            batch_paths[i].push_back(continuous_point);
            batch_coverages[i].push_back(new_state.coverage_map);

            states[i] = new_state;
        }

        if (all_terminated) { 
            std::cout << "step:" << step << std::endl;
            break; }
    }

    // 5. ����������ʧ
    auto total_loss = loss_calculator.compute_batch_loss(batch_paths, batch_coverages,
        vector_fields_tensor_for_loss, region_masks)/* + temp_loss / batch_size*/;//��������ʱ��ʧ

    // 6. ���򴫲�
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
    return coverage_ratio >= 0.95f;  // 95%������Ϊ���
}
torch::Tensor PathPlanningTrainer::update_coverage(const torch::Tensor& coverage_map,
    const torch::Tensor& point_coords) {
    auto new_coverage = coverage_map.clone();
    auto batch_size = coverage_map.size(0);

    for (int b = 0; b < batch_size; ++b) {
        int x = static_cast<int>(point_coords[b][0].item<float>());
        int y = static_cast<int>(point_coords[b][1].item<float>());

        // ����3x3����
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

    // ��ȡ����������
    auto encoder_params = encoder.parameters();
    for (auto& param : encoder_params) {
        all_params.push_back(param);
    }

    // ��ȡ����������
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
// 1. ������ʧ - ����δ��������ı���
torch::Tensor MultiObjectiveLoss::compute_coverage_loss(const torch::Tensor& final_coverage,
    const torch::Tensor& region_mask) {
    // ��8��8���������ϲ�����24��24
    auto highres_mask = upsample_mask_3x3(region_mask).to(device);  // [24, 24]

    // ֻ������Ҫ���������δ���ǲ���
    auto target_area = torch::sum(highres_mask);
    auto covered_area = torch::sum(final_coverage * highres_mask);
    auto uncovered_area = target_area - covered_area;

    // �������
    if (target_area.item<float>() < 1e-6) {
        return torch::zeros({ 1 }, device);
    }

    return uncovered_area / target_area;
}
// 2. �ص���ʧ - ��������ڵ�֮����ص�
torch::Tensor MultiObjectiveLoss::compute_overlap_loss(const std::vector<torch::Tensor>& path_points,
    const std::vector<torch::Tensor>& coverage_maps) {
    if (path_points.size() < 3) {
        return torch::zeros({ 1 }, device);
    }

    torch::Tensor total_overlap = torch::zeros({ 1 }, device);
    int overlap_count = 0;

    // Ϊÿ��������串������
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
        auto current_coverage = point_coverages[i];  // ��ǰ��ĸ�������

        // �ۼ�֮ǰ���з����ڵ�ĸ��������ų�i-1��i-2��
        torch::Tensor previous_coverage = torch::zeros({ 24, 24 }, device);
        for (size_t j = 0; j < i - 2; ++j) {
            previous_coverage += point_coverages[j];
        }
        previous_coverage = torch::clamp(previous_coverage, 0.0f, 1.0f);

        // �����ص�����
        auto overlap = current_coverage * previous_coverage;
        total_overlap += torch::sum(overlap);
        overlap_count++;
    }

    if (overlap_count > 0) {
        return total_overlap / overlap_count;
    }
    return torch::zeros({ 1 }, device);
}
// 3. ƽ����ʧ - ����·��ת�۽Ƕ�
torch::Tensor MultiObjectiveLoss::compute_smoothness_loss(const std::vector<torch::Tensor>& path_points) {
    if (path_points.size() <= 2) {
        return torch::zeros({ 1 }, device);
    }

    torch::Tensor total_angle_loss = torch::zeros({ 1 }, device);
    int angle_count = 0;// ��Ч�нǼ������

    for (size_t i = 2; i < path_points.size(); ++i) {
        // 1. �������������߶ε�����������������Ҫ��
        auto point1 = path_points[i-2].flatten();
        auto point2 = path_points[i - 1].flatten();
        auto point3 = path_points[i].flatten();
        TORCH_CHECK(point1.size(0) == 2 && point2.size(0) == 2 && point3.size(0) == 2,
            "All points must have exactly 2 elements");
        // v1���ӵ�i-2����ָ���i-1���������
        auto v1 = (path_points[i - 1] - path_points[i - 2]).to(device);  // ǰһ������
        // v2���ӵ�i-1����ָ���i���������
        auto v2 = (path_points[i] - path_points[i - 1]).to(device);    // ��ǰ������

        // 2. ����������L2�������߶γ��ȣ����ڹ�һ����
        auto norm_v1 = torch::norm(v1);
        auto norm_v2 = torch::norm(v2);

        // 3.����������
        if (norm_v1.item<float>() < 1e-6 || norm_v2.item<float>() < 1e-6) {
            continue;
        }
        // 4. ����нǵ�����ֵ��cos�� = (v1��v2) / (||v1||��||v2||)��

        auto cos_angle = torch::dot(v1, v2) / (norm_v1 * norm_v2);
        auto angle_loss = 0.5f * (1.0f - cos_angle);  // Ŀ�꣺�ǶȽӽ�0��

        total_angle_loss += angle_loss;
        angle_count++;
    }

    if (angle_count > 0) {
        return total_angle_loss / angle_count;
    }
    return torch::zeros({ 1 }, device);
}
// 4. ��������ʧ - ����·������ʸ�����Ķ���̶�
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

        // ��ȡ��㴦�ĳ�����
        auto field_at_point = bilinear_sample_vector_field(field_on_device, start_point); 
        //std::cout << field_at_point.sizes() << std::endl;
        auto norm_segment = torch::norm(segment_vector);
        auto norm_field = torch::norm(field_at_point);

        // ����������
        if (norm_segment.item<float>() < 1e-6) {
            segment_vector = segment_vector + 1e-6 * torch::randn_like(segment_vector);
            norm_segment = torch::norm(segment_vector);
        }
        if (norm_field.item<float>() < 1e-6) {
            field_at_point = field_at_point + 1e-6 * torch::randn_like(field_at_point);
            norm_field = torch::norm(field_at_point);
        }

        // ���㷽��һ���ԣ�����˳�����泡�����ԣ�
        //auto dot_product = torch::dot(segment_vector, field_at_point);
        //auto cos_angle = dot_product / (norm_segment * norm_field);
        //auto alignment_loss = 1.0f - torch::abs(cos_angle);  // Ŀ�꣺�ǶȽӽ�0���180��
        // ��λ����������
        auto segment_dir = segment_vector / norm_segment;
        auto field_dir = field_at_point / norm_field;

        // ����˳/�泡�����L2����
        auto diff_forward_norm = torch::norm(segment_dir - field_dir);  // ˳������
        auto diff_backward_norm = torch::norm(segment_dir + field_dir); // �泡����

        // ʹ��soft-min���Ӳmin�������ݶȽض�
        auto min_diff = soft_min(diff_forward_norm, diff_backward_norm, 10.0f);

        // ƽ���ͷ���ǿ�ݶ�
        auto alignment_loss = min_diff * min_diff;


        total_alignment_loss += alignment_loss;
        segment_count++;
    }

    if (segment_count > 0) {
        return total_alignment_loss / segment_count;
    }
    return torch::zeros({ 1 }, device);
}
// �����������ϲ�������
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
// ������������ȡ��ĸ�������
torch::Tensor MultiObjectiveLoss::get_point_coverage(const torch::Tensor& point) {
    torch::Tensor point_flat;
    if (point.dim() == 2 && point.size(0) == 1 && point.size(1) == 2) {
        // ��״Ϊ [1, 2] �����
        point_flat = point.squeeze(0);  // ��Ϊ [2]
    }
    else if (point.dim() == 1 && point.size(0) == 2) {
        // ��״Ϊ [2] �����
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

// ����������˫���Բ���ʸ����
torch::Tensor MultiObjectiveLoss::bilinear_sample_vector_field(const torch::Tensor& vector_field,
    const torch::Tensor& point) {
    // ȷ��point����ȷ��״��
    TORCH_CHECK(point.dim() == 1 && point.size(0) == 2,
        "Point must be 1D tensor with 2 elements for sampling");
    TORCH_CHECK(point.requires_grad(), "point δ�����ݶȣ��޷������ݶȣ�");

    // vector_field: [1, 2, 24, 24]��batch=1, channels=2(x/y����), height=24, width=24��
   // ���Ƚ� [24,24,2] תΪ PyTorch ����Ҫ��� [N, C, H, W] ��ʽ
    auto field = vector_field.permute({ 2, 0, 1 }).unsqueeze(0);  // [1, 2, 24, 24]

    // grid_sample Ҫ�����귶Χ��x/y �� [-1, 1]���������ݼ��Ĺ�һ����Χһ�£�
    // point: [2] �� תΪ [1, 1, 1, 2]��N, H, W, 2������ʾ��1�������㡱
    auto grid = point.unsqueeze(0).unsqueeze(0).unsqueeze(0);  // [1,1,1,2]
    grid = (grid / 23.0f) * 2.0f - 1.0f;  // [0,23] �� [-1,1]
    // 3. �ɵ�˫���Բ���
    auto sampled = torch::nn::functional::grid_sample(
        field,
        grid,
        torch::nn::functional::GridSampleFuncOptions()
        .mode(torch::kBilinear)  // ˫���Բ�ֵ
        .padding_mode(torch::kBorder)  // �߽���ԭ��ֵ���
        .align_corners(true)  // ȷ���������������ȷ
    );
    auto result = sampled.squeeze();
  

    return result;
}

//class ImprovedSampler
 // ����������Ӳ��������������
std::pair<torch::Tensor, torch::Tensor> ImprovedSampler::sample_with_temperature_st(const torch::Tensor& point_logits) {
    float current_temp = get_current_temperature();

    // ȷ��logits����̫����softmax����
    auto stabilized_logits = point_logits - torch::max(point_logits);

    // Ӧ���¶�����
    auto tempered_logits = stabilized_logits / current_temp;
    // ����ʶȵ�Gumbel����
    auto gumbel_noise = -torch::log(-torch::log(torch::rand_like(tempered_logits) + 1e-8) + 1e-8);
    auto noisy_logits = tempered_logits + gumbel_noise;

    auto soft_samples = torch::softmax(noisy_logits, -1);
    auto hard_samples = torch::argmax(soft_samples, -1);
    
    // ���Gumbel����
   
    //auto gumbel_noise = -torch::log(-torch::log(torch::rand_like(point_logits)));

    // Ӧ���¶�����
    //auto tempered_logits = (point_logits + gumbel_noise) / current_temp;
    //auto soft_samples = torch::softmax(tempered_logits, -1);

    // Ӳ��������ǰ�򴫲���ʹ��argmax
    //auto hard_samples = torch::argmax(soft_samples, -1);

    // Straight-Through���ɣ���ǰ�򴫲���ʹ��Ӳ�������ڷ��򴫲���ʹ�������
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
// �����ʷֲ�ת��Ϊ��������
torch::Tensor ImprovedSampler::probability_to_continuous_coordinates(const torch::Tensor& probs, int grid_size) {
    auto batch_size = probs.size(0);
    auto coords = torch::zeros({ batch_size, 2 }, probs.device());

    // Ԥ�������п��ܵ�����
    auto grid_coords = torch::zeros({ grid_size * grid_size, 2 }, probs.device());
    for (int i = 0; i < grid_size * grid_size; ++i) {
        grid_coords[i][0] = static_cast<float>(i % grid_size);
        grid_coords[i][1] = static_cast<float>(i / grid_size);
    }

    // �����������꣺probs [batch, 576] �� grid_coords [576, 2] = [batch, 2]
    coords = torch::matmul(probs, grid_coords);
    //std::cout <<"coords:" << coords.sizes() << std::endl;
    return coords;
}

