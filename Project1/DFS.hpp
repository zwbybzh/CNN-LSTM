#include<iostream>
#include<vector>
#include "FieldDataset.hpp"
class dfs {//哈密顿路径，数据量极大，不可求解
private:
    FieldDataset field_maker;
    torch::Tensor all_next_point(torch::Tensor start_point) {
        torch::Tensor next_point=torch::zeros({2});
        

        return next_point;

    }
public:
    dfs() {
        field_maker = FieldDataset(1, 24, 24, false, 0);
    }
    void forward() {
        auto [field,start_point] = field_maker.get(0);//生成场和起始点
        std::cout << "start point: " << start_point << std::endl;//[2]
        std::cout<<"start point: "<<start_point.sizes()<<std::endl;//[x,y]
        std::cout<<"field: "<<field.sizes()<<std::endl;//[W,H,2]
        start_point[0] = 1;
        start_point[1] = 1;
        std::cout << "start point: " << start_point << std::endl;
        all_next_point(start_point);

    }



};
