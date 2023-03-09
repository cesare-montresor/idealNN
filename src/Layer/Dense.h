//
// Created by cesare on 08/02/23.
//

#include <Layer/Layer.h>
#include <Tensor/Tensor.h>

#ifndef IDEALNN_DENSE_H
#define IDEALNN_DENSE_H


//
// Created by Cesare on 25/10/2022.
//

namespace IdealNN {

    /// Implements a standard Dense or Fully Connected Layer.
    struct Dense final: public Layer {
    protected:
        /// Number of input units
        ArraySize in;
        /// Number of output units
        ArraySize out;

    public:
        /// matrix of weights
        TensorRef weights;
        /// vector of biases
        TensorRef bias;

        /// Utility method to create Dense layer objects wrapped in a shared pointer
        /// @param in number of neurons in the previous layer
        /// @param out number of neurons in the next layer
        static DenseRef MakeDense(int in, int out);

        /// Construct a dense layer specifying the number of in/out units
        /// @param in Number of input units
        /// @param out Number of out units
        Dense(ArraySize  in, ArraySize  out);

        /// Accept single item from a batch data and execute a linear transformation for the forward pass.
        /// @param x Tensor representing a single instance of data
        TensorRef forward(TensorRef x) override;

        /// Accept single item from a batch data and execute the backward pass. Gradient formula:
        /// @param dx Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        void backward(TensorRef dx, ArrayIndex i) override;

        /// Return weights and bias Tensors
        TensorArrayRef parameters() override;
    };
}

#endif //IDEALNN_DENSE_H
