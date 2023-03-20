//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_CROSSENTROPYLOSS_H
#define IDEALNN_CROSSENTROPYLOSS_H

#include <Common.h>
#include <Tensor/Tensor.h>
#include <Loss/Loss.h>

namespace IdealNN {
    struct CrossEntropyLoss;
    /// Default type for pointers to CrossEntropy Loss
    using CrossEntropyLossRef = shared_ptr<CrossEntropyLoss>;



    /// Implements the Cross Entropy Loss function, useful for multi-class classification tasks.
    struct CrossEntropyLoss: public Loss {
        /// Utility method to create CrossEntropyLoss objects wrapped in a shared pointer
        static CrossEntropyLossRef MakeCrossEntropyLoss();

        /// Compute the loss function using the formula: \f$ \frac{\sum( y * -\log(hat{y}) )}{batch_size} \f$
        /// @param ys_hat Represent an array containing the ground-truth
        /// @param ys Represent an array containing the ground-truth
        ScalarValue loss(TensorArrayRef ys_hat, TensorArrayRef ys) override;

        /// Triggers the cascade of all backward pass. Gradient formula: \f$  -\frac{y}{hat{y}} \f$
        void backward() override;
    };
}

#endif //IDEALNN_CROSSENTROPYLOSS_H
