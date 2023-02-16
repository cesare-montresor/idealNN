//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#ifndef IDEALNN_LAYER_H
#define IDEALNN_LAYER_H

#include <vector>
#include <memory>
#include "../Tensor/Tensor.h"

namespace IdealNN {
    struct Layer {
        virtual TensorArrayRef forward(TensorArrayRef batch) = 0 ;
        virtual void backward() = 0;
    };
}



#endif //IDEALNN_LAYER_H
