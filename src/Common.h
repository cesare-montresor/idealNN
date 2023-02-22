//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_COMMON_H
#define IDEALNN_COMMON_H

#include <vector>
#include <memory>
#include <Eigen/Eigen>



using namespace std;

namespace IdealNN{
    typedef unsigned long int ArrayIndex;
    typedef unsigned long int ArraySize;

    typedef Eigen::MatrixXf Matrix;
    using MatrixRef = shared_ptr<Matrix>;
    typedef vector<MatrixRef> MatrixArray;
    typedef shared_ptr<MatrixArray> MatrixArrayRef;
    typedef Eigen::internal::traits<Matrix>::Scalar CoeffRef;

    typedef float ScalarValue;
    typedef vector<ScalarValue> ScalarValueArray;
    typedef shared_ptr<ScalarValueArray> ScalarValueArrayRef;
    const ScalarValue ScalarDelta = 0.00001;

    // resolve/avoid circular deps:
    struct Tensor;
    using TensorRef = shared_ptr<Tensor>;
    typedef vector<TensorRef> TensorArray;
    typedef shared_ptr<TensorArray> TensorArrayRef;

    struct Scalar;
    using ScalarRef = shared_ptr<Scalar>;
    typedef vector<ScalarRef> ScalarArray;
    typedef shared_ptr<ScalarArray> ScalarArrayRef;

    struct Layer;
    using LayerRef = shared_ptr<Layer>;
    typedef vector<LayerRef> LayerArray;
    typedef shared_ptr<LayerArray> LayerArrayRef;

    struct Dense;
    using DenseRef = shared_ptr<Dense>;

    struct Loss;
    using LossRef = shared_ptr<Loss>;
}



#endif //IDEALNN_COMMON_H
