//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_MSELOSS_H
#define IDEALNN_MSELOSS_H

#include "../Common.h"
#include "../Tensor/Tensor.h"
#include "Loss.h"

namespace IdealNN {
    struct MSELoss:Loss {
        Scalar loss(TensorArrayRef y, TensorArrayRef y_hat ) override;
    };
}

#endif //IDEALNN_MSELOSS_H
