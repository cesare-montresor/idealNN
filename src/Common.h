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



/// Common.h contains type definitions for the whole project
/// serves also as forward declaration to avoid circular dependencies.

namespace IdealNN{

    /// Default type for Scalar Values
    using ScalarValue = double;

    /// Default type for vector/array index
    using ArrayIndex = std::int64_t;
    /// Default type for vector/array size
    using ArraySize = std::int64_t;

    /// Default type for shared pointers
    template<typename T>
    using shared_ptr = std::shared_ptr<T>;
    //is this hiding too much information about the type of pointer used?
    //using Ref = std::shared_ptr<T>;

    /// Default type for arrays
    template<typename T>
    using vector = std::vector<T>;

    /// Default type for random engine
    using random_engine = std::default_random_engine;
    /// Default type for random engine for normal/gaussian distribution
    using normal_distribution = std::normal_distribution<ScalarValue>;
    /// Default type for random engine for uniform distribution
    using uniform_distribution = std::uniform_real_distribution<ScalarValue>;

    /// Default type for strings
    using string = std::string;

    /// Default type for TensorData
    using TensorData = Eigen::MatrixXd;
    /// Default type for TensorData pointers
    using TensorDataRef = std::shared_ptr<TensorData>;
    /// Default type for TensorData arrays
    using TensorDataArray = vector<TensorDataRef> ;
    /// Default type for pointers to arrays of TensorData
    using TensorDataArrayRef = shared_ptr<TensorDataArray> ;

    // Default type for TensorData coefficients (unused)
    // using Coeff = Eigen::internal::traits<TensorData>::Scalar ;


    /// Default type for arrays of Scalar Values
    using ScalarValueArray = vector<ScalarValue>;
    /// Default type for pointers to arrays of Scalar Values
    using ScalarValueArrayRef = shared_ptr<ScalarValueArray> ;
    /// ScalarDelta defines the tolerance for real value comparison
    const constexpr ScalarValue ScalarDelta = 0.00001;



    /// Forward declaration of Tensor class
    struct Tensor;
    /// Default type for Tensor pointers
    using TensorRef = shared_ptr<Tensor>;
    /// Default type for array of Tensors
    using TensorArray = vector<TensorRef>;
    /// Default type for pointers to array of Tensors
    using TensorArrayRef = shared_ptr<TensorArray>;

    /// Forward declaration of Layer class
    struct Layer;
    /// Default type for Layer pointers
    using LayerRef = shared_ptr<Layer>;
    /// Default type for array of Layers
    using LayerArray = vector<LayerRef>;
    /// Default type for pointers to array of Layers
    using LayerArrayRef = shared_ptr<LayerArray>;

    /// Forward declaration of LinearLayer layer class
    struct LinearLayer;
    /// Default type for pointers to LinearLayer layers
    using DenseRef = shared_ptr<LinearLayer>;

    /// Forward declaration of SigmoidActivation class
    struct SigmoidActivation;
    /// Default type for pointers to Sigmoid Activation
    using SigmoidActivationRef = shared_ptr<SigmoidActivation>;

    /// Forward declaration of SigmoidActivation class
    struct TanhActivation;
    /// Default type for pointers to Sigmoid Activation
    using TanhActivationRef = shared_ptr<TanhActivation>;

    /// Forward declaration of RELUActivationRef layer class
    struct RELUActivation;
    /// Default type for pointers to RELU Activation
    using RELUActivationRef = shared_ptr<RELUActivation>;

    /// Forward declaration of SoftmaxActivation layer class
    struct SoftmaxActivation;
    /// Default type for pointers to Softmax Activation
    using SoftmaxActivationRef = shared_ptr<SoftmaxActivation>;

    /// Forward declaration of Loss class
    struct Loss;
    /// Default type for pointers to Loss error
    using LossRef = shared_ptr<Loss>;
}



#endif //IDEALNN_COMMON_H
