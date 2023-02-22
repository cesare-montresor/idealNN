//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_LOSS_H
#define IDEALNN_LOSS_H

#include "../Common.h"
#include "../Tensor/Tensor.h"

namespace IdealNN {
    struct Loss {
        virtual ScalarValue loss(TensorArrayRef y, TensorArrayRef y_hat )=0;
        virtual void backward()=0;
    };
}

#endif //IDEALNN_LOSS_H
