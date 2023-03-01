//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_SIGMOIDACTIVATION_H
#define IDEALNN_SIGMOIDACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {

    /// Implementation of the Sigmoid activation
    struct SigmoidActivation:  public Activation {

        /// Accept single item from a batch data and execute the forward pass: 1/(1 - e^-x)
        /// @param x Tensor representing a single instance of data
        /// @param i Index of the data inside the mini batch, useful for storing partial results to be reused my the backward pass
        TensorRef forward(TensorRef x, ArrayIndex i) override;

        /// Accept single item from a batch data and execute the forward pass.
        /// @param xd Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        void backward(TensorRef dx, ArrayIndex i) override;
    };

} // IdealNN

#endif //IDEALNN_SIGMOIDACTIVATION_H
