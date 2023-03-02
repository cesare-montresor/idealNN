//
// Created by cesare on 15/02/23.
//

#ifndef IDEALNN_LOSS_H
#define IDEALNN_LOSS_H

#include <Common.h>
#include <Tensor/Tensor.h>

namespace IdealNN {

    /// Virtual class acting as common interface for all Loss functions
    struct Loss {
    protected:
        /// ys_hat holds the values of the predictions made by the model, for the entire mini-batch, store it for the backward pass.
        TensorArrayRef ys_hat;
        /// deltas are the relative errors for each sample of the batch. They are computed in the forward pass and used in the backward pass.
        TensorArrayRef deltas;

    public:
        /// Compute the loss and the deltas, returns the loss and store the gradients for the backward pass.
        /// @param ys_hat Represent an array containing the ground-truth
        /// @param ys Represent an array containing the ground-truth
        virtual ScalarValue loss(TensorArrayRef ys_hats, TensorArrayRef ys )=0;
        /// Triggers the cascade of all backward pass
        virtual void backward()=0;
    };
}

#endif //IDEALNN_LOSS_H
