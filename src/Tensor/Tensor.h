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
        /// Utility method to used to create an empty arrays of Tensors
        static TensorArrayRef MakeTensorArray();
        /// Utility method to used to create an arrays of Tensors by preallocating the size of the array
        /// @param size Size of the array
        static TensorArrayRef MakeTensorArray(ArraySize size);
        /// Utility method to used to move the ownership of an array of Tensors to a pointer
        /// @param tensorArray Layer array to be wrapped by the pointer.
        static TensorArrayRef MakeTensorArray(TensorArray tensorArray);


        //static

        /// Create a pointer to a 2D tensor with given rows and columns
        /// @param rows Number of rows
        /// @param cols Number of columns
        static TensorRef MakeTensor(ArraySize rows, ArraySize cols);

        /// Create a pointer to a Tensor from an existing Tensor
        /// @param tensor Source tensor
        static TensorRef MakeTensor(Tensor &tensor);

        /// Create a Tensor from a pointer to another tensor
        /// @param tensor Pointer to tensor
        static TensorRef MakeTensor(const TensorRef& tensor);

        /// Create a Tensor from a Matrix
        /// @param matrix A matrix object
        static TensorRef MakeTensor(const Matrix &matrix);

        /// Create a Tensor from a pointer to a Matrix
        /// @param matrix Pointer to matrix
        static TensorRef MakeTensor(const MatrixRef& matrix);



        //constructors
        /// Create a 2D tensor with given rows and columns
        /// @param rows Number of rows
        /// @param cols Number of columns
        Tensor(ArraySize rows, ArraySize cols);

        /// Create a Tensor from an existing Tensor
        /// @param tensor Source tensor
        Tensor(Tensor const &tensor);

        /// Create a Tensor from a pointer to another tensor
        /// @param tensor Pointer to tensor
        explicit Tensor(const TensorRef& tensor);

        /// Create a Tensor from a Matrix
        /// @param matrix A matrix object
        explicit Tensor(Matrix const &matrix);

        /// Create a Tensor from a pointer to a Matrix
        /// @param matrix Pointer to matrix
        explicit Tensor(MatrixRef matrix);



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

        /// Initialize the data to zero
        void initZero();

        /// Initialize the data to random values between [-1, 1] using a uniform distribution.
        void initUniform();

        /// Initialize the data to random values between [-1/sqrt(fan_in), 1/sqrt(fan_in)] using a uniform distribution.
        /// @param fan_in Number of input units
        void initKaiming(ArrayIndex fan_in);
    };


}

#endif //IDEALNN_TENSOR_H
