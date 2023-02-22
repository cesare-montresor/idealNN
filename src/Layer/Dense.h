//
// Created by cesare on 08/02/23.
//

#include "Layer.h"
#include "../Tensor/Tensor.h"

#ifndef IDEALNN_DENSE_H
#define IDEALNN_DENSE_H


//
// Created by Cesare on 25/10/2022.
//

namespace IdealNN {


    struct Dense : Layer {


        int in, out;
        TensorArrayRef xs;
        TensorRef weights;
        TensorRef bias;
        TensorRef gradients;
        TensorArrayRef activations;

        Dense(int in, int out);

        TensorArrayRef forward(TensorArrayRef batch) override;

        void backward(TensorArrayRef deltas) override;

    };
}

#endif //IDEALNN_DENSE_H
