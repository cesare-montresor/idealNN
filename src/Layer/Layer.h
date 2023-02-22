//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#ifndef IDEALNN_LAYER_H
#define IDEALNN_LAYER_H

#include <vector>
#include <memory>
#include "../Common.h"



namespace IdealNN {

    struct Layer: public std::enable_shared_from_this<Layer> {
        //static array

        virtual TensorArrayRef forward(TensorArrayRef batch) = 0 ;
        virtual void backward(TensorArrayRef deltas) = 0;
    };
}



#endif //IDEALNN_LAYER_H
