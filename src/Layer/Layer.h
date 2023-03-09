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
    /*  https://en.cppreference.com/w/cpp/memory/enable_shared_from_this
        std::enable_shared_from_this allows an object t that is currently managed by a std::shared_ptr named pt to safely generate additional std::shared_ptr instances pt1, pt2, ... that all share ownership of t with pt.
        Publicly inheriting from std::enable_shared_from_this<T> provides the type T with a member function shared_from_this.
        If an object t of type T is managed by a std::shared_ptr<T> named pt, then calling T::shared_from_this will return a new std::shared_ptr<T> that shares ownership of t with pt.
     */
    //shared pointer multipli, hanno lo stesso owner

    protected:
        /// Array of tensors, used to store the whole batch of inputs. Used in the backward pass.
        TensorArrayRef inputs;
        /// Array of tensors, used to store the whole batch of output. Often used in the backward pass.
        TensorArrayRef outputs;



    public:
        /// Accept a batch of data and call the forward pass on each of them, it also stores the inputs inside inputs
        /// @param xs Array of tensors, representing the a patch of data.
        TensorArrayRef forwardBatch(const TensorArrayRef& xs);

        /// Accept single item from a batch data and execute the forward pass.
        /// @param x Tensor representing a single instance of data
        virtual TensorRef forward(TensorRef x) = 0 ;

        /// Accept single gradient from the previous layer
        /// @param dx Tensor representing the gradiant flowing from the previous layers.
        /// @param i Index of the data inside the mini batch, useful to connect the gradients dx with the original input data.
        virtual void backward(TensorRef dx, ArrayIndex i) = 0;

        /// Return the list of tensors that can be optimized.
        virtual TensorArrayRef parameters() = 0;
    };

}



#endif //IDEALNN_LAYER_H
