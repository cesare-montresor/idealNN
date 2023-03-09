//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_CROSSENTROPYLOSS_H
#define IDEALNN_CROSSENTROPYLOSS_H

#include <Common.h>
#include <Tensor/Tensor.h>
#include <Loss/Loss.h>

namespace IdealNN {
    /// Implements the Cross Entropy Loss function, useful for multi-class classification tasks.
    struct CrossEntropyLoss final: public Loss {

        /// Compute the loss function using the formula: sum( y * -log(y_hat) ) / batch_size
        /// @param ys_hat Represent an array containing the ground-truth
        /// @param ys Represent an array containing the ground-truth
        ScalarValue loss(TensorArrayRef ys_hat, TensorArrayRef ys) override;

        /// Triggers the cascade of all backward pass. Gradient formula: -(y / y_hat)
        void backward() override;
    };
}

#endif //IDEALNN_CROSSENTROPYLOSS_H
