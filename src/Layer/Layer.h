//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#ifndef IDEALNN_LAYER_H
#define IDEALNN_LAYER_H

#include <vector>
#include <memory>
#include <Common.h>



namespace IdealNN {
    /// General interface for Layer objects
    struct Layer: public std::enable_shared_from_this<Layer> {
    protected:
        /// Array of tensors, used to store the whole batch of inputs. Used in the backward pass.
        TensorArrayRef xs;

    public:
        /// Accept a batch of data and call the forward pass on each of them, it also stores the inputs inside xs
        /// @param xs Array of tensors, representing the a patch of data.
        TensorArrayRef forwardBatch(TensorArrayRef xs);

        /// Accept single item from a batch data and execute the forward pass.
        /// @param x Tensor representing a single instance of data
        /// @param i Index of the data inside the mini batch, useful for storing partial results to be reused my the backward pass
        virtual TensorRef forward(TensorRef x, ArrayIndex i) = 0 ;

        /// Accept single gradient from the previous layer
        /// @param xd Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        virtual void backward(TensorRef dx, ArrayIndex i) = 0;

        /// Return the list of tensors that can be optimized.
        virtual TensorArrayRef parameters() = 0;
    };

}



#endif //IDEALNN_LAYER_H
