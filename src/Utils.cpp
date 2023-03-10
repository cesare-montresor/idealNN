//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include "Common.h"
#include "Tensor/Tensor.h"
#include "Layer/Dense.h"
#include "Activation/SigmoidActivation.h"
#include "Activation/RELUActivation.h"
#include "Activation/SoftmaxActivation.h"

namespace IdealNN{
    namespace Utils{

        TensorArrayRef slice(const TensorArrayRef& array, ArrayIndex start, ArraySize count){
            auto as = static_cast<ArraySize>(array->size());

            if(start + 1 >= as ) {return Tensor::MakeTensorArray();}
            auto items_count = start + count < as ? count : (as - start);

            auto first = array->begin() + start;
            auto last = array->begin() + start + items_count;

            return Tensor::MakeTensorArray(TensorArray{first, last});
        }


        bool ScalarValueEqual(ScalarValue a, ScalarValue b){ return a - b < ScalarDelta; }

        ScalarValueArrayRef MakeScalarValueArray() { return std::make_shared<ScalarValueArray>(); }
        ScalarValueArrayRef MakeScalarValueArray(ArraySize size) { return std::make_shared<ScalarValueArray>(size); }
        ScalarValueArrayRef MakeScalarValueArray(ScalarValueArray scalarValueArray) { return std::make_shared<ScalarValueArray>(std::move(scalarValueArray)); }

        MatrixRef MakeMatrix(ArraySize rows, ArraySize cols) { return std::make_shared<Matrix>(rows, cols); }
        MatrixRef MakeMatrix(Matrix matrix) { return std::make_shared<Matrix>(std::move(matrix)); }

        MatrixArrayRef MakeMatrixArray() { return std::make_shared<MatrixArray>(); }
        MatrixArrayRef MakeMatrixArray(ArraySize size) { return std::make_shared<MatrixArray>(size); }
        MatrixArrayRef MakeMatrixArray(MatrixArray matrixArray) { return std::make_shared<MatrixArray>(std::move(matrixArray)); }

        LayerArrayRef MakeLayerArray(){ return std::make_shared<LayerArray>(); }
        LayerArrayRef MakeLayerArray(ArraySize size){ return std::make_shared<LayerArray>(size); }
        LayerArrayRef MakeLayerArray(LayerArray layerArray){ return std::make_shared<LayerArray>(std::move(layerArray)); }


        //TODO: make a log function that can be globally enabled/disabled
        //void log(){
            //https://stackoverflow.com/questions/3649278/how-can-i-get-the-class-name-from-a-c-object
        //}
    }
}

#endif //IDEALNN_UTILS_H

