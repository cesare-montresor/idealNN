//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_TANHACTIVATION_H
#define IDEALNN_TANHACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {
    struct TanhActivation;
    /// Default type for pointers to Sigmoid Activation
    using TanhActivationRef = shared_ptr<TanhActivation>;

    /// Implementation of the Tanh activation
    struct TanhActivation: public Activation {
        /// Utility method to create TanhActivation objects wrapped in a shared pointer
        static TanhActivationRef MakeTanhActivation();

        /// Accept single item from a batch data and execute the forward pass, using the formula: tanh(x)
        /// @param x Tensor representing a single instance of data
        TensorRef forward(TensorRef x) override;

        /// Accept single item from a batch data and execute the backward pass. Gradient formula: ( 1 - tanh(x)^2 )
        /// @param dx Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        void backward(TensorRef dx, ArrayIndex i) override;
    };

} // IdealNN

#endif //IDEALNN_TANHACTIVATION_H
