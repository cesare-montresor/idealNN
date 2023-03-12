//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_MSELOSS_H
#define IDEALNN_MSELOSS_H

#include <Common.h>
#include <Tensor/Tensor.h>
#include <Loss/Loss.h>

namespace IdealNN {
    struct MSELoss;
    /// Default type for pointers to MSELoss error
    using MSELossRef = shared_ptr<MSELoss>;

    /// Implements the Mean Squared Error loss function, useful for regression tasks.
    struct MSELoss: public Loss {
        /// Utility method to create MSELoss objects wrapped in a shared pointer
        static MSELossRef MakeMSELoss();

        /// Compute the loss function using the formula: \f$ \frac{1}{2}*\frac{\sum{(y-\hat{y})^2}}{batch_size} \f$
        /// @param ys_hat Represent an array containing the ground-truth
        /// @param ys Represent an array containing the ground-truth
        ScalarValue loss(TensorArrayRef ys_hat, TensorArrayRef ys ) override;

        /// Triggers the cascade of all backward pass. Gradient formula: \f$ ( \hat{y} - y ) \f$
        void backward() override;
    };
}

#endif //IDEALNN_MSELOSS_H
