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
        static TensorRef MakeTensor(const TensorRef &tensor);

        /// Create a Tensor from a TensorData
        /// @param tensorData A tensorData object
        static TensorRef MakeTensor(const TensorData &tensorData);

        /// Create a Tensor from a pointer to a TensorData
        /// @param tensorData Pointer to tensorData
        static TensorRef MakeTensor(const TensorDataRef &tensorData);



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

        /// Create a Tensor from a TensorData
        /// @param tensorData A tensorData object
        explicit Tensor(const TensorData &tensorData);

        /// Create a Tensor from a pointer to a TensorData
        /// @param tensorData Pointer to tensorData
        explicit Tensor(TensorDataRef tensorData);



        //Properties
        /// Holds the data of the tensor.
        TensorDataRef data;
        /// Defines if the Tensor should accumulate gradients or not.
        bool use_grads = true;
        /// Represent operation executed on the Tensor, it is used by the backward pass. If is not set, it will interrupt the backward steps.
        LayerRef operation;
        /// Defines the last operation executed on the Tensor, it is used by the backward pass.
        TensorDataRef gradients;

        //Methods
        /// Reset the accumulated gradients to zero.
        void zero_grad();
        /// Returns a Tensor pointer with a subset of the data.
        TensorRef view(ArrayIndex row_min, ArrayIndex col_min, ArraySize row_count, ArraySize col_count);

        /// Initialize the data to zero
        void initZero();

        /// Initialize the data to random values between [-1, 1] using a uniform distribution.
        void initUniform();

        /// Initialize the data to random uniform values between [-1*bound, 1*bound], with bound = (gain/sqrt(fan_in))*scaleFactor
        /// @param fan_in Number of input units
        /// @param gain Defined per type of layer ( see more: https://pytorch.org/docs/stable/nn.init.html )
        /// @param scaleFactor Scaling factor: sqrt(3) for weights, 1 for biases
        void initKaiming(ArrayIndex fan_in, ScalarValue gain, ScalarValue scaleFactor);


        //void Tensor::initNormal();
        //see more https://stackoverflow.com/questions/35827926/eigen-matrix-library-filling-a-matrix-with-random-float-values-in-a-given-range

    };


}

#endif //IDEALNN_TENSOR_H
