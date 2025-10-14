#pragma once
#include <torch/torch.h>
#include <torch/optim/schedulers/lr_scheduler.h>
class InputProcessor{
public:
    InputProcessor() {};
    // ����: 8x8�������� + 8x8x2ʸ����
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
    // ��߶�������ȡ
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
    torch::nn::LSTM lstm{ nullptr };// LSTM���壺�������������������״̬
    std::shared_ptr<AttentionLayer> attention{ nullptr };  // ʹ��shared_ptr// ע�����㣺�۽�CNN�����еĹؼ�����
    torch::nn::Linear point_embedding{ nullptr };// ��Ƕ��㣺��2D����תΪ��ά����
    torch::nn::Linear output_layer{ nullptr };// ����㣺Ԥ����һ����ĸ��ʷֲ�
    torch::nn::Linear coverage_embedding{ nullptr };// ����Ƕ��㣺������ͼתΪ��ά����
    torch::nn::Linear feature_projection{ nullptr };// ����ͶӰ�㣺ƥ��CNN������ע����ά��
    int hidden_size;// LSTM��״̬ά�ȣ�Ĭ��512��
    int m_layer_num;
public:
    int get_m_layer_num() {
        return m_layer_num;
    }
    int get_hidden_size();
    LSTMPathDecoder(int hidden_size = 512,int layer_num = 2) ;

    struct DecoderState {
        torch::Tensor hidden_state; //LSTM��״̬[2, batch, hidden_size]��2�㣩
        torch::Tensor cell_state;// LSTMϸ��״̬ [2, batch, hidden_size]
        torch::Tensor coverage_map;  // ��ǰ����״̬ [batch, 24, 24]
        torch::Tensor last_point;    // ��һ�������� [batch, 2]
        int step_count = 0;// ��ǰ���������ڿ�����󳤶ȣ�

        // ���캯��
        DecoderState(const torch::Tensor& region_mask, int batch_size = 1, int hidden_size = 512, int num_layers = 2) {
            // ��ʼ������״̬��ϸ��״̬
             // ��region_mask��ȡ�ߴ���Ϣ
            int64_t original_height = region_mask.size(0);
            int64_t original_width = region_mask.size(1);
            int64_t highres_height = original_height * 3;  // �Ź����ϲ���
            int64_t highres_width = original_width * 3;

            hidden_state = torch::zeros({ num_layers, batch_size, hidden_size });  // 2��LSTM
            cell_state = torch::zeros({ num_layers, batch_size, hidden_size });

            coverage_map = initialize_coverage_from_mask(region_mask, batch_size);//����������������ʼ������

            last_point = torch::full({ batch_size, 2 },1.0f);  // ��ʼ����Ϊ(0.5,0.5)
            step_count = 0;
        }
    
        // ����region_mask��ʼ������ͼ
        torch::Tensor initialize_coverage_from_mask(const torch::Tensor& region_mask, int batch_size) {
            int64_t h = region_mask.size(0);
            int64_t w = region_mask.size(1);
            int64_t highres_h = h * 3;
            int64_t highres_w = w * 3;

            auto coverage = torch::zeros({ batch_size, highres_h, highres_w });

            // ���region_mask��ֵ������������ʼ������״̬
            // ���磺�Ǵ�ӡ�����ʼ��Ϊ�Ѹ���
            if (batch_size == 1) {
                for (int i = 0; i < h; ++i) {
                    for (int j = 0; j < w; ++j) {
                        if (region_mask[i][j].item<float>() < 0.5f) {
                            // �Ǵ�ӡ���򣬱��Ϊ�Ѹ���
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
        //std::cout << "initialize_batch_states ���յ��豸: " << device << std::endl;
        // Ϊ�����е�ÿ����������������״̬
        for (int i = 0; i < batch_size; ++i) {
            DecoderState state(region_masks[i], 1, hidden_size,layer_num);  // ÿ������batch_size=1

            // �ƶ����豸
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
        DecoderState state,                  // ������״̬
        const torch::Tensor& vector_field    // [batch, 24, 24, 2] ʸ����
    );
//private:
     torch::Tensor create_coverage_mask(const torch::Tensor& coverage_map);
    // ������������������ת��Ϊ����
    static torch::Tensor idx_to_coordinate(const torch::Tensor& indices, int grid_size = 24);
    // �������������¸���״̬
    static torch::Tensor update_coverage(const torch::Tensor& coverage_map,
        const torch::Tensor& point_coords);
    
};

//ģ���¶��˻��softmax
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
    // ����1���¶��˻��Gumbel-Softmax���Ƽ���
    torch::Tensor sample_with_temperature(const torch::Tensor& point_logits);
private:
public:
    
    float get_current_temperature() {
        // ָ���˻�����ѵ�������𽥽����¶�
        float temp = initial_temperature * std::pow(decay_rate, training_step);
        return std::max(temp, min_temperature);
    }
    // �����ʷֲ�ת��Ϊ��������
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
    

    // ѵ������
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
    // ������ʧ����ӿ�
    torch::Tensor compute_batch_loss(
        const std::vector<std::vector<torch::Tensor>>& batch_paths,      // [batch_size][path_length, 2]
        const std::vector<std::vector<torch::Tensor>>& batch_coverages,  // [batch_size][path_length, 24, 24]
        const torch::Tensor& vector_fields,                              // [batch_size, 24, 24, 2]
        const std::vector<torch::Tensor>& region_masks);
    // ����������ʧ����
    torch::Tensor compute_sample_loss(
        const std::vector<torch::Tensor>& path_points,      // [path_length, 2]
        const std::vector<torch::Tensor>& coverage_maps,    // [path_length, 24, 24]  
        const torch::Tensor& vector_field,                  // [24, 24, 2]
        const torch::Tensor& region_mask);
private:
    // 1. ������ʧ - ����δ��������ı���
    torch::Tensor compute_coverage_loss(const torch::Tensor& final_coverage,
        const torch::Tensor& region_mask);
    // 2. �ص���ʧ - ��������ڵ�֮����ص�
    torch::Tensor compute_overlap_loss(const std::vector<torch::Tensor>& path_points,
        const std::vector<torch::Tensor>& coverage_maps);

    // 3. ƽ����ʧ - ����·��ת�۽Ƕ�
    torch::Tensor compute_smoothness_loss(const std::vector<torch::Tensor>& path_points);
    // 4. ��������ʧ - ����·������ʸ�����Ķ���̶�
    torch::Tensor compute_alignment_loss(const std::vector<torch::Tensor>&path_points,
        const torch::Tensor & vector_field);
    // �����������ϲ�������
    torch::Tensor upsample_mask_3x3(const torch::Tensor& mask);
    // ������������ȡ��ĸ�������
    torch::Tensor get_point_coverage(const torch::Tensor& point);
    // ����������˫���Բ���ʸ����
    torch::Tensor bilinear_sample_vector_field(const torch::Tensor& vector_field,
        const torch::Tensor& point);

};
//ģ���¶��˻��softmax
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
//    // ����1���¶��˻��Gumbel-Softmax���Ƽ���
//    torch::Tensor sample_with_temperature(const torch::Tensor& point_logits);
//private:
//    float get_current_temperature() {
//        // ָ���˻�����ѵ�������𽥽����¶�
//        float temp = initial_temperature * std::pow(decay_rate, training_step);
//        return std::max(temp, min_temperature);
//    }
//};