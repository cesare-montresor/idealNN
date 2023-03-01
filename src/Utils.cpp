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

        TensorArray slice(TensorArray array, ArrayIndex start, ArraySize count){
            if(start + 1 >= array.size() ) {return TensorArray{};}
            auto items_count = start + count < array.size() ? count : (array.size() - start);

            auto first = array.begin() + start;
            auto last = array.begin() + start + (ArraySize)items_count;

            return TensorArray{first, last};
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


        //static array
        TensorArrayRef MakeTensorArray(){ return std::make_shared<TensorArray>(); }
        TensorArrayRef MakeTensorArray(ArraySize size){ return std::make_shared<TensorArray>(size); }
        TensorArrayRef MakeTensorArray(TensorArray tensorArray){ return std::make_shared<TensorArray>(std::move(tensorArray)); }

        //static array
        ScalarArrayRef MakeScalarArray(){ return std::make_shared<ScalarArray>(); }
        ScalarArrayRef MakeScalarArray(ArraySize size){ return std::make_shared<ScalarArray>(size); }
        ScalarArrayRef MakeScalarArray(ScalarArray scalarArray){ return std::make_shared<ScalarArray>(std::move(scalarArray)); }

        //
        DenseRef MakeDense(int in, int out) { return std::make_shared<Dense>(in, out); }
        SigmoidActivationRef MakeSigmoidActivation() { return std::make_shared<SigmoidActivation>(); }
        RELUActivationRef MakeRELUActivation() { return std::make_shared<RELUActivation>(); }
        SoftmaxActivationRef MakeSoftmaxActivation() { return std::make_shared<SoftmaxActivation>(); }
    };
}

#endif //IDEALNN_UTILS_H

