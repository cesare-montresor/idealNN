//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_MSELOSS_H
#define IDEALNN_MSELOSS_H

#include <Common.h>
#include <Tensor/Tensor.h>
#include <Loss/Loss.h>

namespace IdealNN {
    /// Implements the Mean Squared Error loss function, useful for regression tasks.
    struct MSELoss: public Loss {
        /// Compute the loss function using the formula: sum(1/2 * (y - y_hat)^2) / batch_size
        /// @param ys Represent an array containing the ground-truth
        /// @param ys_hat Represent an array containing the ground-truth
        ScalarValue loss(TensorArrayRef ys, TensorArrayRef ys_hat ) override;
        /// Triggers the cascade of all backward pass
        void backward() override;
    };
}

#endif //IDEALNN_MSELOSS_H
