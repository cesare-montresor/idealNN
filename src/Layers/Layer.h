//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#ifndef IDEALNN_LAYER_H
#define IDEALNN_LAYER_H

#include <vector>

struct Layer{
    std::vector<int> *dims;

    Layer();
    double forward();
    double backward();
};


#endif //IDEALNN_LAYER_H
