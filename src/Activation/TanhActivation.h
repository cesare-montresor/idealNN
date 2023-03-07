//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_SIGMOIDACTIVATION_H
#define IDEALNN_SIGMOIDACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {

    /// Implementation of the Tanh activation
    struct TanhActivation:  public Activation {

        /// Accept single item from a batch data and execute the forward pass, using the formula: tanh(x)
        /// @param x Tensor representing a single instance of data
        /// @param i Index of the data inside the mini batch, useful for storing partial results to be reused by the backward pass
        TensorRef forward(TensorRef x, ArrayIndex i) override;

        /// Accept single item from a batch data and execute the backward pass. Gradient formula: ( 1 - tanh(x)^2 )
        /// @param xd Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        void backward(TensorRef dx, ArrayIndex i) override;
    };

} // IdealNN

#endif //IDEALNN_SIGMOIDACTIVATION_H
