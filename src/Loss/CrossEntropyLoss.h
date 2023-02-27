//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_CROSSENTROPYLOSS_H
#define IDEALNN_CROSSENTROPYLOSS_H

#include <Common.h>
#include <Tensor/Tensor.h>
#include <Loss/Loss.h>

namespace IdealNN {
    struct CrossEntropyLoss: public Loss {
        TensorArrayRef ys;
        TensorArrayRef deltas;

        ScalarValue loss(TensorArrayRef y, TensorArrayRef y_hat ) override;
        void backward() override;
    };
}

#endif //IDEALNN_CROSSENTROPYLOSS_H
