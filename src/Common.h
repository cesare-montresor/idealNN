//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_COMMON_H
#define IDEALNN_COMMON_H

#include <vector>
#include <memory>
#include <cstdint>
#include <Eigen/Eigen>
#include <iostream>
#include <random>
//#include <optional>

//NOTE: unsupported => provided "as is" (https://eigen.tuxfamily.org/dox/unsupported/index.html)
//#include <unsupported/Eigen/MatrixFunctions>

namespace IdealNN{

    // Wrappers for standard types, to IdealNN types
    using ArrayIndex = std::int64_t;
    using ArraySize = std::int64_t;

    template<typename T>
    using shared_ptr = std::shared_ptr<T>;

    template<typename T>
    using vector = std::vector<T>;

    /*
    template<typename T>
    using optional = std::optional<T>;
    */

    using default_random_engine = std::default_random_engine;
    using string = std::string;

    using Eigen::MatrixXf;
    typedef MatrixXf Matrix;
    using MatrixRef = std::shared_ptr<Matrix>;
    typedef vector<MatrixRef> MatrixArray;
    typedef shared_ptr<MatrixArray> MatrixArrayRef;
    typedef Eigen::internal::traits<Matrix>::Scalar CoeffRef;

    typedef float ScalarValue;
    typedef vector<ScalarValue> ScalarValueArray;
    typedef shared_ptr<ScalarValueArray> ScalarValueArrayRef;
    const ScalarValue ScalarDelta = 0.00001;

    // forward declarations to avoid circular deps:
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

    struct SigmoidActivation;
    using SigmoidActivationRef = shared_ptr<SigmoidActivation>;

    struct RELUActivation;
    using RELUActivationRef = shared_ptr<RELUActivation>;

    struct SoftmaxActivation;
    using SoftmaxActivationRef = shared_ptr<SoftmaxActivation>;

    struct Dense;
    using DenseRef = shared_ptr<Dense>;

    struct Loss;
    using LossRef = shared_ptr<Loss>;
}



#endif //IDEALNN_COMMON_H
