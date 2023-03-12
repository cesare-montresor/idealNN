//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_RELUACTIVATION_H
#define IDEALNN_RELUACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {
    struct RELUActivation;
    /// Default type for pointers to RELU Activation
    using RELUActivationRef = shared_ptr<RELUActivation>;

    /// Implementation of the RELU activation
    struct RELUActivation:  public Activation {
        /// Utility method to create RELUActivation objects wrapped in a shared pointer
        static RELUActivationRef MakeRELUActivation();

        /// Accept single item from a batch data and execute the forward pass using the formula: max(0,x)
        /// @param x Tensor representing a single instance of data
        TensorRef forward(TensorRef x) override;

        /// Accept single item from a batch data and execute the backward pass. Gradient formula:
        /// @param dx Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        void backward(TensorRef dx, ArrayIndex i) override;
    };

} // IdealNN

#endif //IDEALNN_RELUACTIVATION_H
