//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include <Common.h>

namespace IdealNN::Utils{

    /// Method used to get portions of array of tensors
    /// @param array Array of tensors
    /// @param start Index of the first item to be returned
    /// @param count Number of items to be returned
    /// The method always returns an array of tensors, eventually empty.
    TensorArrayRef slice(const TensorArrayRef &array, ArrayIndex start, ArraySize count);

    /// Method used to verify the equality of two real numbers, it uses ScalarDelta (see Common.h) to define a tolerance gap.
    /// @param a A real value
    /// @param b A real value
    bool ScalarValueEqual(ScalarValue a, ScalarValue b);

    /// Cast vector size size_type (aka unsigned long) to standard ArraySize
    /// @param vector A vector
    template<typename T>
    ArraySize getSize(const vector<T> &vector){
        auto size = static_cast<ArraySize>(vector.size());
        if(size < 0){
            auto msg = "[IdealNN::Utils::slice] Conversion from unsigned to signed to did overflow.";
            throw std::runtime_error(msg);
        }
        return size;
    }

    /// Cast vector size size_type (aka unsigned long) to standard ArraySize
    /// @param vector Pointer to Vector
    template<typename T>
    ArraySize getSize(const shared_ptr<vector<T>> &vector){
        return getSize((*vector));
    }


    /// Utility method to used to create an empty arrays of ScalarValues
    ScalarValueArrayRef MakeScalarValueArray();
    /// Utility method to used to create an arrays of ScalarValues by preallocating the size of the array
    /// @param size Size of the array
    ScalarValueArrayRef MakeScalarValueArray(ArraySize size);
    /// Utility method to used to move the ownership of ScalarValuesArray to a pointer
    /// @param scalarValueArray ScalarValueArray to be wrapped by the pointer.
    ScalarValueArrayRef MakeScalarValueArray(ScalarValueArray scalarValueArray);

    /// Utility method to used to create a TensorData of given size
    /// @param rows Number of rows of the TensorData
    /// @param cols Number of columns of the TensorData
    TensorDataRef MakeTensorData(ArraySize rows, ArraySize cols);
    /// Utility method to used to move the ownership of TensorData to a pointer
    /// @param tensorData TensorData to be wrapped by the pointer.
    TensorDataRef MakeTensorData(TensorData &&tensorData);

    /// Utility method to used to create an empty arrays of Matrices
    TensorDataArrayRef MakeTensorDataArray();
    /// Utility method to used to create an arrays of Matrices by preallocating the size of the array
    /// @param size Size of the array
    TensorDataArrayRef MakeTensorDataArray(ArraySize size);
    /// Utility method to used to move the ownership of an array of Matrices to a pointer
    /// @param tensorDataArray TensorDataArray to be wrapped by the pointer.
    TensorDataArrayRef MakeTensorDataArray(TensorDataArray tensorDataArray);


    /// Utility method to used to create an empty arrays of Layers
    LayerArrayRef MakeLayerArray();
    /// Utility method to used to create an arrays of Layers by preallocating the size of the array
    /// @param size Size of the array
    LayerArrayRef MakeLayerArray(ArraySize size);
    /// Utility method to used to move the ownership of an array of Layers to a pointer
    /// @param layerArray Layer array to be wrapped by the pointer.
    LayerArrayRef MakeLayerArray(LayerArray layerArray);







    //TODO: Utility function used to produce output
    //void Log();
}

#endif //IDEALNN_UTILS_H

