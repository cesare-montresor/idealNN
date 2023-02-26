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

        MatrixArray slice(MatrixArray array, ArrayIndex start, ArraySize count){
            if(start + 1 >= array.size() ) {return MatrixArray();}
            auto items_count = start + count < array.size() ? count : (array.size() - start);

            auto first = array.begin() + start;
            auto last = array.begin() + start + items_count;

            return MatrixArray{first, last};
        }

        TensorArray slice(TensorArray array, ArrayIndex start, ArraySize count){
            if(start + 1 >= array.size() ) {return TensorArray();}
            auto items_count = start + count < array.size() ? count : (array.size() - start);

            auto first = array.begin() + start;
            auto last = array.begin() + start + items_count;

            return TensorArray{first, last};
        }


        MatrixRef MakeMatrix(ArraySize in, ArraySize out) { return make_shared<Matrix>(in, out); }

        bool Equal(ScalarValue a, ScalarValue b){
            return a - b < ScalarDelta;
        }


        LayerArrayRef MakeLayerArray(){ return make_shared<LayerArray>(); }
        LayerArrayRef MakeLayerArray(ArraySize size){ return make_shared<LayerArray>(size); }
        LayerArrayRef MakeLayerArray(LayerArray layerArray){ return make_shared<LayerArray>(layerArray); }


        //static array
        TensorArrayRef MakeTensorArray(){ return make_shared<TensorArray>(); }
        TensorArrayRef MakeTensorArray(ArraySize size){ return make_shared<TensorArray>(size); }
        TensorArrayRef MakeTensorArray(TensorArray tensorArray){ return make_shared<TensorArray>(tensorArray); }

        //static array
        ScalarArrayRef MakeScalarArray(){ return make_shared<ScalarArray>(); }
        ScalarArrayRef MakeScalarArray(ArraySize size){ return make_shared<ScalarArray>(size); }
        ScalarArrayRef MakeScalarArray(ScalarArray scalarArray){ return make_shared<ScalarArray>(scalarArray); }

        //
        DenseRef MakeDense(int in, int out) { return make_shared<Dense>(in, out); }
        SigmoidActivationRef MakeSigmoidActivation() { return make_shared<SigmoidActivation>(); }
        RELUActivationRef MakeRELUActivation() { return make_shared<RELUActivation>(); }
        SoftmaxActivationRef MakeSoftmaxActivation() { return make_shared<SoftmaxActivation>(); }
    };
}

#endif //IDEALNN_UTILS_H

