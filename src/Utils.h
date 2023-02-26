//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include "Common.h"

namespace IdealNN{
    namespace Utils{

        //TODO: template vector<T>
        MatrixArray slice(MatrixArray array, ArrayIndex start, ArraySize count);
        TensorArray slice(TensorArray array, ArrayIndex start, ArraySize count);

        MatrixRef MakeMatrix(ArraySize in, ArraySize out);

        LayerArrayRef MakeLayerArray();
        LayerArrayRef MakeLayerArray(ArraySize size);
        LayerArrayRef MakeLayerArray(LayerArray tensorArray);

        TensorArrayRef MakeTensorArray();
        TensorArrayRef MakeTensorArray(ArraySize size);
        TensorArrayRef MakeTensorArray(TensorArray tensorArray);

        ScalarArrayRef MakeScalarArray();
        ScalarArrayRef MakeScalarArray(ArraySize size);
        ScalarArrayRef MakeScalarArray(ScalarArray scalarArray);

        DenseRef MakeDense(int in, int out);
        SigmoidActivationRef MakeSigmoidActivation();
        RELUActivationRef MakeRELUActivation();

        bool Equal(ScalarValue a, ScalarValue b);
    };
}

#endif //IDEALNN_UTILS_H

