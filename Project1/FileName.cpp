#include "FieldDataset.hpp"
#include "Net.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <torch/torch.h>
#include <random>
#include <opencv2/opencv.hpp>
struct TrainingConfig {
    int num_epochs = 100;
    int batch_size = 4;
    int num_samples = 1000;
    float learning_rate = 3e-3;
    std::string model_save_path = "trained_model.pt";
    bool use_cuda = true;
};

class TrainingMonitor {
private:
    std::vector<float> train_losses;
    std::vector<float> epoch_times;

public:
    void record_epoch(float loss, float time_ms) {
        train_losses.push_back(loss);
        epoch_times.push_back(time_ms);
    }

    void print_training_summary() {
        if (train_losses.empty()) return;

        float avg_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f) / train_losses.size();
        float avg_time = std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0f) / epoch_times.size();

        std::cout << "\n=== Training Summary ===" << std::endl;
        std::cout << "Average loss: " << avg_loss << std::endl;
        std::cout << "Average epoch time: " << avg_time << "ms" << std::endl;
        std::cout << "Final loss: " << train_losses.back() << std::endl;
    }

    // ���ӻ��ӿڣ�Ԥ����
    void plot_training_curves() {
        std::cout << "[Visualization] Training curves would be plotted here" << std::endl;
        // δ�����Լ���matplotlib���������ӻ���
    }
};

class SimpleTrainer {
    private :
    TrainingConfig config;
    torch::Device device;
    PathPlanningTrainer trainer;
    std::unique_ptr<FieldDataset> dataset;
    TrainingMonitor monitor;//��ʱûɶ��
    std::mt19937 rng;//ͬ����ʱûɶ��

    float best_loss = std::numeric_limits<float>::max();
    std::string best_model_path = "best_model";

public:
    SimpleTrainer(const TrainingConfig& cfg) : config(cfg), device(torch::kCUDA) {
        // ��ʼ�������������
        std::random_device rd;
        rng.seed(rd());//��ʱû��
        // �����豸
        if (config.use_cuda && torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using CUDA device" << std::endl;
        }
        else {
            device = torch::kCPU;
            std::cout << "Using CPU device" << std::endl;
        }

        // ��ʼ��ѵ����
        trainer = PathPlanningTrainer(device);

        // �������ݼ�
        dataset = std::make_unique<FieldDataset>(config.num_samples, 8, 8);

        std::cout << "Trainer initialized with:" << std::endl;
        std::cout << "  Epochs: " << config.num_epochs << std::endl;
        std::cout << "  Batch size: " << config.batch_size << std::endl;
        std::cout << "  Samples: " << config.num_samples << std::endl;
        std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    }

    // ��������������루ȫ1����ʾ����������Ҫ��䣩
    torch::Tensor generate_region_mask() {
        return torch::ones({ 8, 8 }, torch::kFloat32);
        //return torch::rand({ 8, 8 }, torch::kFloat32).gt(0.5).to(torch::kFloat32);
    }

    // ��ȡѵ������
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
        get_training_batchs(int batch_size) {
        std::vector<torch::Tensor> region_masks;
        std::vector<torch::Tensor> vector_fields;

        for (int i = 0; i < batch_size; ++i) {
            // ������ɵĳ�
            auto example = dataset->get(i);
            torch::Tensor vector_field = example.data;
            //torch::Tensor vector_field = torch::zeros({ 8, 8, 2 });
            //�̶���
            //for (int i = 0; i < 8; ++i) {
            //    for (int j = 0; j < 8; ++j) {
            //        float x = (j - 4.0f) / 4.0f;
            //        float y = (i - 4.0f) / 4.0f;
            //        float radius = std::sqrt(x * x + y * y);
            //        if (radius < 0.1f) radius = 0.1f;

            //        // ��ת�������߷���
            //        vector_field[i][j][0] = -y / radius;
            //        vector_field[i][j][1] = x / radius;
            //    }
            //}

            // �����������루ȫ1��
            auto region_mask = generate_region_mask();

            region_masks.push_back(region_mask);
            vector_fields.push_back(vector_field);
        }

        return { region_masks, vector_fields };
    }
    // ����ѵ��
    void train() {
        std::cout << "\n=== Starting Training ===" << std::endl;

        int total_batches = config.num_samples / config.batch_size;
        if (total_batches == 0) total_batches = 1;
        auto scheduler = torch::optim::StepLR(*trainer.optimizer, /*step_size=*/2, /*gamma=*/0.8);
        auto [region_masks, vector_fields] = get_training_batchs(config.num_samples);


        for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            float epoch_loss = 0.0f;
            int batches_processed = 0;
            int sample_num = 0;
            // ѵ��һ��epoch
            for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
                try {
                    std::cout << "batch_idx:" << (batch_idx + 1) * config.batch_size << std::endl;
                    // ��ȡѵ�������������
                    std::vector <torch::Tensor> region_masks_epoch;
                    std::vector <torch::Tensor> vector_fields_epoch;

                    int top = sample_num;

                    for (sample_num; sample_num < std::min((top + config.batch_size), config.num_samples); sample_num++) {
                        region_masks_epoch.push_back(region_masks[sample_num]);
                        vector_fields_epoch.push_back(vector_fields[sample_num]);
                    }

                    //��֤���ݼ�


                    // ִ��ѵ������
                    auto loss = trainer.train_step(region_masks_epoch, vector_fields_epoch,
                        config.batch_size, 8, 8);

                    epoch_loss += loss.item<float>();
                    batches_processed++;

                    // ÿ1�����δ�ӡһ�ν���
                    if (batch_idx % 1 == 0) {
                        std::cout << "Epoch " << epoch + 1 << ", Batch " << batch_idx + 1
                            << "/" << total_batches << ", Loss: " << loss.item<float>() << std::endl;
                    }

                }
                catch (const std::exception& e) {
                    std::cerr << "Error in batch " << batch_idx + 1 << ": " << e.what() << std::endl;
                    continue;
                }
            }


            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                epoch_end - epoch_start);

            // ����ƽ����ʧ
            float avg_loss = batches_processed > 0 ? epoch_loss / batches_processed : 0.0f;

            std::cout << "Epoch " << epoch << " completed. "
                << "Avg Loss: " << avg_loss
                << ", Time: " << epoch_duration.count() << "ms" << std::endl;

            if (avg_loss < best_loss) {
                best_loss = avg_loss;
                save_best_model(epoch, avg_loss);
            }
            if (epoch % 10 == 0) {
                save_checkpoint(epoch, avg_loss);
            }
            scheduler.step();
            for (const auto& group : trainer.optimizer->param_groups()) {
                std::cout << "��ǰѧϰ��: " << group.options().get_lr() << std::endl;
            }

        }

        std::cout << "=== Training Completed ===" << std::endl;
        //monitor.print_training_summary();//��ӡ����ģ�Ͳ���

        std::cout << "Best loss achieved: " << best_loss << std::endl;
    }
    // ������㣨���ֲ��䣩
    void save_checkpoint(int epoch, float loss) {
        std::cout << "Checkpoint: Epoch " << epoch << ", Loss: " << loss << std::endl;
    }
    void save_best_model(int epoch, float loss) {
        try {
            // ʹ�����л�����鵵���������
            {
                torch::serialize::OutputArchive archive;
                trainer.encoder.save(archive);
                archive.save_to("best_model_encoder.pt");
            }

            // ʹ�����л�����鵵���������
            {
                torch::serialize::OutputArchive archive;
                trainer.decoder.save(archive);
                archive.save_to("best_model_decoder.pt");
            }
            std::cout << ">>> BEST MODEL SAVED! Epoch: " << epoch
                << ", Loss: " << loss
                << ", Path: " << best_model_path << "_*.pt" << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "Error saving best model: " << e.what() << std::endl;
        }
    }
    //void validate_model(int epoch) {
    //    std::cout << "\n--- Validating Model (Epoch " << epoch << ") ---" << std::endl;

    //    try {
    //        // ʹ��С����������֤
    //        auto [val_masks, val_fields] = get_training_batch(2);
    //        auto loss = trainer.train_step(val_masks, val_fields, 2, 8, 8);

    //        std::cout << "Validation loss: " << loss.item<float>() << std::endl;

    //        // ���ӻ��ӿڣ�Ԥ����
    //        visualize_validation_results(epoch, loss.item<float>());

    //    }
    //    catch (const std::exception& e) {
    //        std::cerr << "Validation failed: " << e.what() << std::endl;
    //    }
    //}
    // �������
    
   
    // ���ӻ���֤�����Ԥ���ӿڣ�
    void visualize_validation_results(int epoch, float loss) {
        std::cout << "[Visualization] Epoch " << epoch << " validation loss: " << loss << std::endl;
        // δ���������·�����ӻ�������ͼ���ӻ���
    }
    // ����ģ��
    void test() {
        std::cout << "\n=== Testing Model ===" << std::endl;

        // ���ɲ�������
        auto [test_masks, test_fields] = get_training_batchs(2);

        try {
            // ����һ��ѵ������������
            auto loss = trainer.train_step(test_masks, test_fields, 4, 8, 8);
            std::cout << "Test completed. Loss: " << loss.item<float>() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
        }
    }
};

class MinimalTest {
public:
    void run() {
        std::cout << "=== Minimal Test ===" << std::endl;

        // ʹ��GPU���в���
        torch::Device device = torch::kCUDA;

        try {
            // ����ѵ����
            PathPlanningTrainer trainer(device);
            std::cout << "Trainer created successfully" << std::endl;

            // ������������
            std::vector<torch::Tensor> region_masks;
            std::vector<torch::Tensor> vector_fields;

            for (int i = 0; i < 2; ++i) {
                region_masks.push_back(torch::ones({ 8, 8 }, torch::kFloat32));
                vector_fields.push_back(torch::randn({ 8, 8, 2 }));
            }

            std::cout << "Test data created" << std::endl;

            // ���е���ѵ������
            for (auto& mask : region_masks) {
                mask = mask.to(device);  // �Ƶ��� states ��ͬ���豸
            }
            for (auto& field : vector_fields) {
                field = field.to(device);  // �Ƶ��� states ��ͬ���豸
            }
            std::cout << "train_step_start:" << std::endl;
            auto loss = trainer.train_step(region_masks, vector_fields, 2, 8, 8);
            std::cout << "Training step completed. Loss: " << loss.item<float>() << std::endl;

            std::cout << "=== Minimal Test PASSED ===" << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "=== Minimal Test FAILED ===" << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
};

static void run_() {
    std::cout << "3D Printing Path Planning Training Framework" << std::endl;
    std::cout << "============================================" << std::endl;

    MinimalTest minimal_test;
    minimal_test.run();
    std::cout << "Program completed." << std::endl;
}

static void run_2() {
    std::cout << "3D Printing Path Planning Training Framework" << std::endl;
    std::cout << "============================================" << std::endl;
    TrainingConfig config;
    config.num_epochs = 50;           // ����epoch�������ڲ���
    config.batch_size = 64;            // С������С
    config.num_samples = 64;         // ��������
    config.learning_rate = 1e-4;
    config.use_cuda = true;

    SimpleTrainer trainer(config);

    trainer.train();

    std::cout << "Program completed." << std::endl;
}

// ·����������
class PathGenerator {
private:
    CNNEncoder encoder;
    LSTMPathDecoder decoder;
    InputProcessor input_processor;
    torch::Device device;
    ImprovedSampler sampler;  // ��Ӳ�����

public:
    PathGenerator(torch::Device device = torch::kCPU) : device(device) {

        encoder.to(device);
        decoder.to(device);
        
    }
    
    // ����ģ��
    bool load_model(const std::string& model_path = "best_model") {
        try {
            {
                torch::serialize::InputArchive archive;
                archive.load_from(model_path + "_encoder.pt");
                encoder.load(archive);
            }
            {
                torch::serialize::InputArchive archive;
                archive.load_from(model_path + "_decoder.pt");
                decoder.load(archive);
            }

            // ����Ϊ����ģʽ
            encoder.eval();
            decoder.eval();

            std::cout << "Model loaded successfully!" << std::endl;
            return true;

        }
        catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

    // ����·��
    std::vector<torch::Tensor> generate_path(
        const torch::Tensor& region_mask,    // [8, 8] ��������
        const torch::Tensor& vector_field,   // [8, 8, 2] ʸ����
        int max_steps = 300                  // ���·������
    ) {
        std::vector<torch::Tensor> path_points;

        try {
            // ȷ����������ȷ���豸��
            auto region_mask_device = region_mask.to(device);
            auto vector_field_device = vector_field.to(device);

            // 1. Ԥ��������
            auto processed_input = input_processor.process(region_mask_device, vector_field_device);
            processed_input = processed_input.unsqueeze(0);  // ���batchά�� [1, 24, 24, 5]
            processed_input = processed_input.to(device);

            // 2. ��ȡCNN����
            auto cnn_features = encoder.forward(processed_input);  // [1, 36, 256]

            // 3. ��ʼ��������״̬
            std::vector<torch::Tensor> masks = { region_mask_device };
            auto states = LSTMPathDecoder::initialize_batch_states(masks, device, decoder.get_hidden_size(),decoder.get_m_layer_num());
            states[0].coverage_map = LSTMPathDecoder::update_coverage(states[0].coverage_map, states[0].last_point);
            torch::Tensor start = torch::ones({ 2 }, torch::device(device));//start point.last_point
            path_points.push_back(start);
            auto state = states[0];  // ȡ��һ��״̬
            //std::cout << "state[0]" << states[0].hidden_state.sizes() << std::endl;
            // 4. ����·������
            for (int step = 0; step < max_steps; ++step) {
                // ����Ƿ���ɸ���
                if (is_coverage_complete(state.coverage_map, region_mask_device)) {
                    std::cout << "Coverage complete at step " << step << std::endl;
                    break;
                }

                // ������ǰ�򴫲�
                auto [point_logits, new_state] = decoder.forward(
                    cnn_features, state, vector_field_device.unsqueeze(0));

                //// ѡ�������ߵĵ㣨̰��������
                //auto next_point_idx = torch::argmax(point_logits, -1);
                //auto next_point = LSTMPathDecoder::idx_to_coordinate(next_point_idx);

                // ʹ���¶Ȳ���������̰������ - ��ѵ������һ��
                //std::cout << point_logits.sizes() << std::endl;
                //point_logits = point_logits.squeeze(0);����ӿ�̫sb�ˣ������⣬��Ҫ��
                auto [straight_through_samples, hard_samples] =
                    sampler.sample_with_temperature_st(point_logits);
                auto next_point_idx = hard_samples;//Ӳ����
                //next_point_idx = next_point_idx.squeeze(0);
               // std::cout <<"next_point_idx.sizes() " << next_point_idx.sizes() << std::endl;
                auto next_point = LSTMPathDecoder::idx_to_coordinate(next_point_idx);
                //std::cout <<"next_point.sizes()" << next_point.sizes() << std::endl;
                // ����״̬
                new_state.coverage_map = LSTMPathDecoder::update_coverage(
                    state.coverage_map, next_point);
                new_state.last_point = next_point;
                state = new_state;

                // ����·���㣨�Ƶ�CPU�Ա��������
                path_points.push_back(next_point.squeeze(0).cpu());

                // ��ӡ����
                if (step % 50 == 0) {
                    auto point_cpu = next_point.squeeze(0).cpu();
                    std::cout << "Step " << step << ": Generated point ("
                        << point_cpu[0].item<float>() << ", "
                        << point_cpu[1].item<float>() << ")" << std::endl;
                }

            }

            std::cout << "Path generation completed. Total points: "
                << path_points.size() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "Error during path generation: " << e.what() << std::endl;
        }

        return path_points;
    }
    void visualize_path_with_field(
        const std::vector<torch::Tensor>& path_points,
        const torch::Tensor& vector_field,
        const torch::Tensor& region_mask,
        const std::string& output_path = "path_visualization.png",
        int cell_size = 20,  // ÿ�����ӵ����ش�С
        int margin = 50      // ͼ��߾�
    ) {
        try {
            // �������
            const int grid_size = 24;  // 24x24 ����
            const int img_width = grid_size * cell_size + 2 * margin;
            const int img_height = grid_size * cell_size + 2 * margin;

            // ������ɫ����ͼ��
            cv::Mat img(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));

            // 1. ����������
            draw_grid(img, grid_size, cell_size, margin);

            // 2. �����������루�����Ҫ��
            draw_region_mask(img, region_mask, grid_size, cell_size, margin);

            // 3. ����ʸ����
            draw_vector_field(img, vector_field, grid_size, cell_size, margin);

            // 4. ����·��
            draw_path(img, path_points, grid_size, cell_size, margin);

            // 5. ���Ƹ���״̬����ѡ��
            draw_coverage_status(img, path_points, grid_size, cell_size, margin);

            // 6. ���ͼ������Ϣ
            draw_legend(img, path_points, output_path, margin);

            // ����ͼ��
            cv::imwrite(output_path, img);
            std::cout << "Visualization saved to: " << output_path << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "Error in visualization: " << e.what() << std::endl;
        }
    }
    private:
        // ����������
        void draw_grid(cv::Mat& img, int grid_size, int cell_size, int margin) {
            cv::Scalar grid_color(200, 200, 200);  // ǳ��ɫ
            cv::Scalar thick_grid_color(150, 150, 150);  // ���ɫ

            // ����ϸ������
            for (int i = 0; i <= grid_size; ++i) {
                int x = margin + i * cell_size;
                int y = margin + i * cell_size;

                // ��ֱ��
                cv::line(img,
                    cv::Point(x, margin),
                    cv::Point(x, margin + grid_size * cell_size),
                    grid_color, 1);

                // ˮƽ��
                cv::line(img,
                    cv::Point(margin, y),
                    cv::Point(margin + grid_size * cell_size, y),
                    grid_color, 1);
            }

            // ���ƴ������ߣ�ÿ4��һ����
            for (int i = 0; i <= grid_size; i += 4) {
                int x = margin + i * cell_size;
                int y = margin + i * cell_size;

                // ��ֱ��
                cv::line(img,
                    cv::Point(x, margin),
                    cv::Point(x, margin + grid_size * cell_size),
                    thick_grid_color, 2);

                // ˮƽ��
                cv::line(img,
                    cv::Point(margin, y),
                    cv::Point(margin + grid_size * cell_size, y),
                    thick_grid_color, 2);
            }

            // ���������ǩ
            for (int i = 0; i < grid_size; i += 4) {
                std::string label = std::to_string(i);
                int x = margin + i * cell_size + 5;
                int y = margin - 10;
                int y_bottom = margin + grid_size * cell_size + 20;

                // ������ǩ
                cv::putText(img, label, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

                // �ײ���ǩ
                cv::putText(img, label, cv::Point(x, y_bottom),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

                // ����ǩ
                cv::putText(img, label, cv::Point(margin - 25, margin + i * cell_size + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

                // �Ҳ��ǩ
                cv::putText(img, label, cv::Point(margin + grid_size * cell_size + 5, margin + i * cell_size + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);
            }
        }

        // ������������
        void draw_region_mask(cv::Mat& img, const torch::Tensor& region_mask,
            int grid_size, int cell_size, int margin) {
            // �ϲ����������뵽24x24
            InputProcessor processor;
            auto highres_mask = processor.upsample_3x3(region_mask).squeeze(-1);

            // ������Ҫ��������
            for (int y = 0; y < grid_size; ++y) {
                for (int x = 0; x < grid_size; ++x) {
                    if (highres_mask[y][x].item<float>() > 0.5f) {
                        int px = margin + x * cell_size;
                        int py = margin + y * cell_size;

                        // ��͸����ɫ������ʾ��Ҫ��������
                        cv::Mat overlay;
                        img.copyTo(overlay);
                        cv::rectangle(overlay,
                            cv::Point(px, py),
                            cv::Point(px + cell_size, py + cell_size),
                            cv::Scalar(255, 200, 200), -1);  // ǳ��ɫ���
                        cv::addWeighted(overlay, 0.3, img, 0.7, 0, img);
                    }
                }
            }
        }

        // ����ʸ����
        void draw_vector_field(cv::Mat& img, const torch::Tensor& vector_field,
            int grid_size, int cell_size, int margin) {
            // ʸ����Ӧ���� [24, 24, 2]
            TORCH_CHECK(vector_field.sizes().size() == 3 &&
                vector_field.size(0) == grid_size &&
                vector_field.size(1) == grid_size &&
                vector_field.size(2) == 2,
                "Vector field must be [24, 24, 2]");

            // �������ʸ���������ڹ�һ��
            float max_length = 0;
            for (int y = 0; y < grid_size; ++y) {
                for (int x = 0; x < grid_size; ++x) {
                    auto vec = vector_field[y][x];
                    float length = std::sqrt(vec[0].item<float>() * vec[0].item<float>() +
                        vec[1].item<float>() * vec[1].item<float>());
                    max_length = std::max(max_length, length);
                }
            }
            if (max_length < 1e-6) max_length = 1.0f;

            // ����ʸ����ͷ
            for (int y = 0; y < grid_size; y += 1) {  // ���Ե���Ϊÿ2��һ����ͷ
                for (int x = 0; x < grid_size; x += 1) {
                    auto vec = vector_field[y][x];
                    float vx = vec[0].item<float>();
                    float vy = vec[1].item<float>();

                    // ��һ��������
                    float length = std::sqrt(vx * vx + vy * vy);
                    if (length < 1e-6) continue;

                    vx = vx / max_length * (cell_size * 0.4f);  // ���ŵ����Ӵ�С��40%
                    vy = vy / max_length * (cell_size * 0.4f);

                    int center_x = margin + x * cell_size + cell_size / 2;
                    int center_y = margin + y * cell_size + cell_size / 2;

                    int end_x = center_x + static_cast<int>(vx);
                    int end_y = center_y + static_cast<int>(vy);

                    // ���Ƽ�ͷ
                    cv::arrowedLine(img,
                        cv::Point(center_x, center_y),
                        cv::Point(end_x, end_y),
                        cv::Scalar(100, 100, 100),  // ��ɫ��ͷ
                        1, cv::LINE_AA, 0, 0.3);
                }
            }
        }

        // ����·��
        void draw_path(cv::Mat& img, const std::vector<torch::Tensor>& path_points,
            int grid_size, int cell_size, int margin) {
            if (path_points.empty()) return;

            // ����·����
            for (size_t i = 1; i < path_points.size(); ++i) {
                auto pt1 = path_points[i - 1];
                auto pt2 = path_points[i];

                float x1 = pt1[0].item<float>();
                float y1 = pt1[1].item<float>();
                float x2 = pt2[0].item<float>();
                float y2 = pt2[1].item<float>();

                // ת��Ϊ��������
                int px1 = margin + static_cast<int>(x1 * cell_size + cell_size / 2);
                int py1 = margin + static_cast<int>(y1 * cell_size + cell_size / 2);
                int px2 = margin + static_cast<int>(x2 * cell_size + cell_size / 2);
                int py2 = margin + static_cast<int>(y2 * cell_size + cell_size / 2);

                // ����·���ε���ɫ�����Ը��ݶεķ����˳��仯��
                cv::Scalar color = get_path_color(i, path_points.size());

                // ����·����
                cv::line(img, cv::Point(px1, py1), cv::Point(px2, py2),
                    color, 2, cv::LINE_AA);
            }

            // ����·����
            for (size_t i = 0; i < path_points.size(); ++i) {
                auto pt = path_points[i];
                float x = pt[0].item<float>();
                float y = pt[1].item<float>();

                int px = margin + static_cast<int>(x * cell_size + cell_size / 2);
                int py = margin + static_cast<int>(y * cell_size + cell_size / 2);

                // �����յ�������
                if (i == 0) {
                    // ��� - ��ɫ
                    cv::circle(img, cv::Point(px, py), 5, cv::Scalar(0, 255, 0), -1);
                    cv::circle(img, cv::Point(px, py), 5, cv::Scalar(0, 100, 0), 2);
                }
                else if (i == path_points.size() - 1) {
                    // �յ� - ��ɫ
                    cv::circle(img, cv::Point(px, py), 5, cv::Scalar(0, 0, 255), -1);
                    cv::circle(img, cv::Point(px, py), 5, cv::Scalar(0, 0, 100), 2);
                }
                else {
                    // �м�� - ��ɫ
                    cv::circle(img, cv::Point(px, py), 3, cv::Scalar(255, 0, 0), -1);
                }

                // ÿ20���������
                if (i % 20 == 0) {
                    cv::putText(img, std::to_string(i),
                        cv::Point(px + 8, py - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 0, 0), 1);
                }
            }
        }

        // ���Ƹ���״̬
        void draw_coverage_status(cv::Mat& img, const std::vector<torch::Tensor>& path_points,
            int grid_size, int cell_size, int margin) {
            // ����ÿ�����ӱ����ǵĴ���
            std::vector<std::vector<int>> coverage_count(grid_size, std::vector<int>(grid_size, 0));

            for (const auto& point : path_points) {
                int x = static_cast<int>(point[0].item<float>());
                int y = static_cast<int>(point[1].item<float>());

                // ����3x3����ĸ��Ǽ���
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size) {
                            coverage_count[ny][nx]++;
                        }
                    }
                }
            }

            // ���Ƹ�������ͼ
            int max_coverage = 1;
            for (const auto& row : coverage_count) {
                for (int count : row) {
                    max_coverage = std::max(max_coverage, count);
                }
            }

            for (int y = 0; y < grid_size; ++y) {
                for (int x = 0; x < grid_size; ++x) {
                    if (coverage_count[y][x] > 0) {
                        int px = margin + x * cell_size;
                        int py = margin + y * cell_size;

                        // ���ݸ��Ǵ���������ɫǿ��
                        float intensity = static_cast<float>(coverage_count[y][x]) / max_coverage;
                        cv::Scalar color(0, static_cast<int>(200 * intensity),
                            static_cast<int>(255 * (1 - intensity)));

                        // ���ư�͸����������
                        cv::Mat overlay;
                        img.copyTo(overlay);
                        cv::rectangle(overlay,
                            cv::Point(px, py),
                            cv::Point(px + cell_size, py + cell_size),
                            color, -1);
                        cv::addWeighted(overlay, 0.3, img, 0.7, 0, img);

                        // ��ʾ���Ǵ���
                        if (coverage_count[y][x] > 1) {
                            cv::putText(img, std::to_string(coverage_count[y][x]),
                                cv::Point(px + 2, py + 12),
                                cv::FONT_HERSHEY_SIMPLEX, 0.3,
                                cv::Scalar(0, 0, 0), 1);
                        }
                    }
                }
            }
        }

        // ����ͼ������Ϣ
        void draw_legend(cv::Mat& img, const std::vector<torch::Tensor>& path_points,
            const std::string& output_path, int margin) {
            int text_y = margin / 2;

            // ����
            cv::putText(img, "3D Printing Path Planning Visualization",
                cv::Point(margin, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            text_y += 25;

            // ·����Ϣ
            std::string path_info = "Path Points: " + std::to_string(path_points.size());
            cv::putText(img, path_info,
                cv::Point(margin, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

            text_y += 20;

            // ͼ��
            int legend_x = img.cols - 150;
            int legend_y = margin;

            // ���ͼ��
            cv::circle(img, cv::Point(legend_x, legend_y), 5, cv::Scalar(0, 255, 0), -1);
            cv::putText(img, "Start", cv::Point(legend_x + 10, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

            legend_y += 20;

            // �յ�ͼ��
            cv::circle(img, cv::Point(legend_x, legend_y), 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(img, "End", cv::Point(legend_x + 10, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

            legend_y += 20;

            // ·��ͼ��
            cv::line(img, cv::Point(legend_x - 5, legend_y), cv::Point(legend_x + 5, legend_y),
                cv::Scalar(255, 0, 0), 2);
            cv::putText(img, "Path", cv::Point(legend_x + 10, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

            legend_y += 20;

            // ����ͼ��
            cv::arrowedLine(img, cv::Point(legend_x - 5, legend_y), cv::Point(legend_x + 5, legend_y),
                cv::Scalar(100, 100, 100), 1);
            cv::putText(img, "Field", cv::Point(legend_x + 10, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }

        // ��ȡ·����ɫ������·���ε�λ�ý��䣩
        cv::Scalar get_path_color(int segment_index, int total_segments) {
            // ����ɫ���䵽��ɫ
            float ratio = static_cast<float>(segment_index) / total_segments;
            int blue = static_cast<int>(255 * (1 - ratio));
            int red = static_cast<int>(255 * ratio);
            return cv::Scalar(blue, 100, red);  // BGR��ʽ
        }

private:
    bool is_coverage_complete(const torch::Tensor& coverage_map,
        const torch::Tensor& region_mask) {
        auto highres_mask = input_processor.upsample_3x3(region_mask).squeeze(-1);
        auto target_area = torch::sum(highres_mask);
        //������������������
        auto binary_coverage = (coverage_map > 0).to(torch::kFloat);
        auto covered_area = torch::sum(binary_coverage * highres_mask);
        float coverage_ratio = covered_area.item<float>() / target_area.item<float>();
        return coverage_ratio >= 0.95f;
    }
};

void run_inference_example() {
    std::cout << "=== Running Inference ===" << std::endl;

    // �����豸
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << device << std::endl;

    // ����·��������
    PathGenerator generator(device);

    // ����ģ��
    if (!generator.load_model("best_model")) {
        std::cerr << "Failed to load model!" << std::endl;
        return;
    }

    // ������������
    auto region_mask = torch::ones({ 8, 8 }, torch::kFloat32);  // ȫ������Ҫ���

    // ����ʸ�����������Ǿ��ȳ�����ת���ȣ�
    auto vector_field = torch::zeros({ 8, 8, 2 }, torch::kFloat32);
    
    // ʾ��1�����ȳ������ң�
    //for (int i = 0; i < 8; ++i) {
    //    for (int j = 0; j < 8; ++j) {
    //        vector_field[i][j][0] = 1.0f;  // x����
    //        vector_field[i][j][1] = 0.0f;  // y����
    //    }
    //}

    // ʾ��2����ת��
    
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float x = (j - 4.0f) / 4.0f;
            float y = (i - 4.0f) / 4.0f;
            float radius = std::sqrt(x*x + y*y);
            if (radius < 0.1f) radius = 0.1f;

            // ��ת�������߷���
            vector_field[i][j][0] = -y / radius;
            vector_field[i][j][1] = x / radius;
        }
    }
    

    // ����·��
    auto path_points = generator.generate_path(region_mask, vector_field);
    InputProcessor  bac;
    auto vector_field_in=bac.upsample_vector_field(vector_field);//������24*24
    // ���·��ͳ����Ϣ
    std::cout << "\n=== Path Statistics ===" << std::endl;
    std::cout << "Total points: " << path_points.size() << std::endl;
    if (!path_points.empty()) {
        auto start_point = path_points[0];
        std::cout <<"start_point" << start_point.sizes() << std::endl;
        auto end_point = path_points.back();
        std::cout << "Start point: (" << start_point[0].item<float>()
            << ", " << start_point[1].item<float>() << ")" << std::endl;
        std::cout << "End point: (" << end_point[0].item<float>()
            << ", " << end_point[1].item<float>() << ")" << std::endl;
    }
    generator.visualize_path_with_field(path_points, vector_field_in, region_mask,
        "path_visualization.png", 25, 60);
    






    // ����·�����ݵ��ļ�����ѡ��
    //save_path_to_file(path_points, "generated_path.txt");
}
//void save_path_to_file(const std::vector<torch::Tensor>& path_points,
//    const std::string& filename) {
//    std::ofstream file(filename);
//    if (!file.is_open()) {
//        std::cerr << "Failed to open file: " << filename << std::endl;
//        return;
//    }
//
//    file << "X,Y" << std::endl;
//    for (const auto& point : path_points) {
//        file << point[0].item<float>() << "," << point[1].item<float>() << std::endl;
//    }
//    file.close();
//    std::cout << "Path data saved to: " << filename << std::endl;
//}






int main() {
    run_2();
   // run_inference_example();
    return 0;
}


// //�������ģ����������
//void load_best_model_for_inference(PathPlanningTrainer& trainer, const std::string& model_path = "best_model") {
//    try {
//        {
//            torch::serialize::InputArchive archive;
//            archive.load_from("best_model_encoder.pt");
//            trainer.encoder.load(archive);
//        }
//        {
//            torch::serialize::InputArchive archive;
//            archive.load_from("best_model_decoder.pt");
//            trainer.decoder.load(archive);
//        }
//
//        // ����Ϊ����ģʽ
//        trainer.encoder.eval();
//        trainer.decoder.eval();
//
//        std::cout << "Best model loaded successfully for inference!" << std::endl;
//
//    }
//    catch (const std::exception& e) {
//        std::cerr << "Error loading best model: " << e.what() << std::endl;
//    }
//}