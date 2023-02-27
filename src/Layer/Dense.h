//
// Created by cesare on 08/02/23.
//

#include <Layer/Layer.h>
#include <Tensor/Tensor.h>

#ifndef IDEALNN_DENSE_H
#define IDEALNN_DENSE_H


//
// Created by Cesare on 25/10/2022.
//

namespace IdealNN {


    struct Dense : public Layer {


        int in, out;
        TensorArrayRef xs;
        TensorRef weights;
        TensorRef bias;
        TensorArrayRef activations;

        Dense(int in, int out);

        TensorArrayRef forwardBatch(TensorArrayRef xs) override;
        TensorRef forward(TensorRef x, ArrayIndex i) override;
        void backward(TensorRef dx, ArrayIndex i) override;
        TensorArrayRef parameters() override;
    };
}

#endif //IDEALNN_DENSE_H
