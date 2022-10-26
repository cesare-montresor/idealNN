//
// Created by Cesare on 25/10/2022.
//

#ifndef IDEALNN_LAYER_HPP
#define IDEALNN_LAYER_HPP

#include <vector>

struct layer {
    std::vector<int> *dims;
    layer();
    double forward();
};


#endif //IDEALNN_LAYER_HPP
