//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include <Common.h>

namespace IdealNN{

    /// Utils namespace is a collection of utility methods to create IdealNN standard objects
    namespace Utils{
        /// Method used to get portions of array of tensors
        /// @param array Array of tensors
        /// @param start Index of the first item to be returned
        /// @param count Number of items to be returned
        /// The method always returns an array of tensors, eventually empty.
        TensorArray slice(TensorArray array, ArrayIndex start, ArraySize count);

        /// Method used to verify the equality of two real numbers, it uses ScalarDelta (see Common.h) to define a tolerance gap.
        /// @param a A real value
        /// @param b A real value
        bool ScalarValueEqual(ScalarValue a, ScalarValue b);

        /// Cast vector size size_type (aka unsigned long) to standard ArraySize
        /// @param size Vector size
        template<typename T>
        ArraySize getSize( shared_ptr<vector<T>> vector){ return static_cast<ArraySize>(vector->size()); }
        template<typename T>
        ArraySize getSize( vector<T> vector){ return static_cast<ArraySize>(vector.size()); }

        /// Utility method to used to create an empty arrays of ScalarValues
        ScalarValueArrayRef MakeScalarValueArray();
        /// Utility method to used to create an arrays of ScalarValues by preallocating the size of the array
        /// @param size Size of the array
        ScalarValueArrayRef MakeScalarValueArray(ArraySize size);
        /// Utility method to used to move the ownership of ScalarValuesArray to a pointer
        /// @param scalarValueArray ScalarValueArray to be wrapped by the pointer.
        ScalarValueArrayRef MakeScalarValueArray(ScalarValueArray scalarValueArray);

        /// Utility method to used to create a matrix of given size
        /// @param rows Number of rows of the Matrix
        /// @param cols Number of columns of the Matrix
        MatrixRef MakeMatrix(ArraySize rows, ArraySize cols);
        /// Utility method to used to move the ownership of Matrix to a pointer
        /// @param matrix Matrix to be wrapped by the pointer.
        MatrixRef MakeMatrix(Matrix matrix);

        /// Utility method to used to create an empty arrays of Matrices
        MatrixArrayRef MakeMatrixArray();
        /// Utility method to used to create an arrays of Matrices by preallocating the size of the array
        /// @param size Size of the array
        MatrixArrayRef MakeMatrixArray(ArraySize size);
        /// Utility method to used to move the ownership of an array of Matrices to a pointer
        /// @param matrixArray Matrix to be wrapped by the pointer.
        MatrixArrayRef MakeMatrixArray(MatrixArray matrixArray);


        /// Utility method to used to create an empty arrays of Layers
        LayerArrayRef MakeLayerArray();
        /// Utility method to used to create an arrays of Layers by preallocating the size of the array
        /// @param size Size of the array
        LayerArrayRef MakeLayerArray(ArraySize size);
        /// Utility method to used to move the ownership of an array of Layers to a pointer
        /// @param layerArray Layer array to be wrapped by the pointer.
        LayerArrayRef MakeLayerArray(LayerArray layerArray);


        /// Utility method to used to create an empty arrays of Tensors
        TensorArrayRef MakeTensorArray();
        /// Utility method to used to create an arrays of Tensors by preallocating the size of the array
        /// @param size Size of the array
        TensorArrayRef MakeTensorArray(ArraySize size);
        /// Utility method to used to move the ownership of an array of Tensors to a pointer
        /// @param tensorArray Layer array to be wrapped by the pointer.
        TensorArrayRef MakeTensorArray(TensorArray tensorArray);


        /// Utility method to used to create an empty arrays of Scalars
        ScalarArrayRef MakeScalarArray();
        /// Utility method to used to create an arrays of Scalars by preallocating the size of the array
        /// @param size Size of the array
        ScalarArrayRef MakeScalarArray(ArraySize size);
        /// Utility method to used to move the ownership of an array of Scalars to a pointer
        /// @param scalarArray Layer array to be wrapped by the pointer.
        ScalarArrayRef MakeScalarArray(ScalarArray scalarArray);

        /// Utility method to create Dense layer objects wrapped in a shared pointer
        /// @param in number of neurons in the previous layer
        /// @param out number of neurons in the next layer
        DenseRef MakeDense(int in, int out);
        /// Utility method to create SigmoidActivation objects wrapped in a shared pointer
        SigmoidActivationRef MakeSigmoidActivation();
        /// Utility method to create RELUActivation objects wrapped in a shared pointer
        RELUActivationRef MakeRELUActivation();
        /// Utility method to create SoftmaxActivation objects wrapped in a shared pointer
        SoftmaxActivationRef MakeSoftmaxActivation();

        /// Utility function used to produce output
        void Log();
    };
}

#endif //IDEALNN_UTILS_H

