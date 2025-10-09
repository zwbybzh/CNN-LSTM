#include "FieldDataset.hpp"
#include "Net.hpp"
#include <iostream>
void test01() {
    FieldDataset dataset(4, 8, 8);  // 样本、宽、高
    torch::data::Example<torch::Tensor, torch::Tensor> data = dataset.get(0);

    torch::Tensor field = data.data;//8*8*2
    torch::Tensor start_point = data.target;//{2}

    auto field_x = field.index({ "...", 0 });//8*8

    torch::Tensor full_ones_mask = torch::ones_like(field_x);
    std::cout << "field:" << field.sizes() << std::endl;
    std::cout << "full_ones_mask:" << full_ones_mask.sizes() << std::endl;
    InputProcessor processor;
    torch::Tensor test = processor.process(full_ones_mask, field);//mask,field

    std::cout << "test:" << test.sizes() << std::endl;
}
void test03(torch::Tensor& h_end);
void test04(torch::Tensor& h_end, torch::Tensor& field);
void test02() {//测试卷积层
    std::vector<torch::Tensor>samples;
    for (int i = 0; i < 4; i++) {//4为样本数
        FieldDataset dataset(4, 8, 8);
        torch::data::Example<torch::Tensor, torch::Tensor> data = dataset.get(0);
        InputProcessor processor;
        torch::Tensor test = processor.process(torch::ones_like(data.data.index({ "...", 0 })), data.data);
        samples.push_back(test);
    }
    torch::Tensor batch1 = torch::stack(samples, 0);
    //std::cout << batch1.sizes() << std::endl;
    CNNEncoder cnn;
    torch::Tensor h_end = cnn.forward(batch1);
    test04(h_end, batch1);
    //std::cout << h_end.sizes() << std::endl;
}
void test03(torch::Tensor& h_end) {//测试lstm注意力层
    torch::Tensor foo = torch::randn({ 4,512 }, torch::dtype(torch::kFloat32));
    AttentionLayer a(512);
    torch::Tensor lstm_end = a.forward(foo, h_end);
    //std::cout << "lstm_end:" << lstm_end << std::endl;
}
void test04(torch::Tensor& h_end, torch::Tensor& field) {
    LSTMPathDecoder decoder(512);
    LSTMPathDecoder::DecoderState state(field);
    for (int step = 0; step < 10; ++step) {
        auto [point_logits, new_state] = decoder.forward(h_end, state, field);
        auto probs = torch::softmax(point_logits, -1);//概率分布中归一化
        auto next_point_idx = torch::multinomial(probs, 1).squeeze(-1);//概率选择点!!避免陷入局部最优
        auto next_point_coords = decoder.idx_to_coordinate(next_point_idx); // 形状 [batch, 2]
        auto new_coverage = decoder.update_coverage(state.coverage_map, next_point_coords);//更新覆盖图
        state = new_state;
        state.last_point = next_point_coords;
        state.coverage_map = new_coverage;
        std::cout << "Step " << step << ": Selected points = " << next_point_coords << std::endl;
    }

}