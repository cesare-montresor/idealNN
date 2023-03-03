//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H

#include <Common.h>



namespace IdealNN {

    /// General purpose Tensor class, currently supporting only 1D and 2D tensors.
    /// Tensors can store also gradients and operations for computing partial derivatives.
    /// As general rule, unless explicitly specified, data is never copied, just referenced for performance reasons.
    struct Tensor {

        //static
        static TensorRef MakeTensor(ArraySize in, ArraySize out);
        static TensorRef MakeTensor(Tensor &tensor);
        static TensorRef MakeTensor(const TensorRef& tensor);
        static TensorRef MakeTensor(const Matrix &matrix);
        static TensorRef MakeTensor(const MatrixRef& matrix);



        //constructors
        /// Create a 2D tensor with given rows and columns
        Tensor(ArraySize rows, ArraySize cols);
        /// Create a Tensor from an existing Tensor
        Tensor(Tensor const &tensor);
        /// Create a Tensor from a pointer to another tensor
        explicit Tensor(const TensorRef& tensor);
        /// Create a Tensor from a pointer to a Matrix
        explicit Tensor(MatrixRef matrix);
        /// Create a Tensor from a Matrix
        explicit Tensor(Matrix const &matrix);


        //Properties
        /// Holds the data of the tensor.
        MatrixRef data;
        /// Defines if the Tensor should accumulate gradients or not.
        bool use_grads = true;
        /// Represent operation executed on the Tensor, it is used by the backward pass. If is not set, it will interrupt the backward steps.
        LayerRef operation;
        /// Defines the last operation executed on the Tensor, it is used by the backward pass.
        MatrixRef gradients;

        //Methods
        /// Reset the accumulated gradients to zero.
        void zero_grad();
        /// Returns a Tensor pointer with a subset of the data.
        TensorRef view(ArrayIndex row_min, ArrayIndex col_min, ArraySize row_count, ArraySize col_count);
    };


}

#endif //IDEALNN_TENSOR_H
