//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include <Common.h>

namespace IdealNN{

    /*
    template<typename T>
    using Ref = std::shared_ptr<T>;
    */

    namespace Utils{

        template<typename T>
        inline std::vector<T> slice(std::shared_ptr<vector<T>> value, ArrayIndex start, ArraySize count) {
            if(start + 1 >= value.size() ) {return T();}
            auto items_count = start + count < value.size() ? count : (value.size() - start);

            auto first = value.begin() + start;
            auto last = value.begin() + start + items_count;

            return std::vector<T>{first, last};
        }

        //TODO: template vector<T>
        MatrixArray slice(MatrixArray array, ArrayIndex start, ArraySize count);
        TensorArray slice(TensorArray array, ArrayIndex start, ArraySize count);

        bool ScalarValueEqual(ScalarValue a, ScalarValue b);

        ScalarValueArrayRef MakeScalarValueArray();
        ScalarValueArrayRef MakeScalarValueArray(ArraySize size);
        ScalarValueArrayRef MakeScalarValueArray(ScalarValueArray scalarValueArray);

        MatrixRef MakeMatrix(ArraySize in, ArraySize out);
        MatrixRef MakeMatrix(Matrix matrix);

        MatrixArrayRef MakeMatrixArray();
        MatrixArrayRef MakeMatrixArray(ArraySize size);
        MatrixArrayRef MakeMatrixArray(LayerArray tensorArray);

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
        SoftmaxActivationRef MakeSoftmaxActivation();

    };
}

#endif //IDEALNN_UTILS_H

