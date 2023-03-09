//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_SIGMOIDACTIVATION_H
#define IDEALNN_SIGMOIDACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {

    /// Implementation of the Sigmoid activation
    struct SigmoidActivation final:  public Activation {
        /// Utility method to create SigmoidActivation objects wrapped in a shared pointer
        static SigmoidActivationRef MakeSigmoidActivation();

        /// Accept single item from a batch data and execute the forward pass, using the formula: 1/(1 - e^-x)
        /// @param x Tensor representing a single instance of data
        TensorRef forward(TensorRef x) override;

        /// Accept single item from a batch data and execute the backward pass. Gradient formula: ( 1 / ( 1 + sigmoid(x) ) )
        /// @param dx Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        void backward(TensorRef dx, ArrayIndex i) override;


    };

} // IdealNN

#endif //IDEALNN_SIGMOIDACTIVATION_H
