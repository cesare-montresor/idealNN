//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_COMMON_H
#define IDEALNN_COMMON_H

/*! @mainpage
 IdealNN is a simple Neural Network framework written in C++ that aims to provide pytorch-like APIs.
 The library is mainly intended for educational purposes to demystify the complexities behind neural network frameworks.

 

<h3>Quick links</h3>
<ul>
 <li><a href="https://github.com/cesare-montresor/idealNN">GitHub IdealNN</a></li>
 <li><a href="https://cesare-montresor.github.io/idealNN/_common_8h_source.html">IdealNN Types</a></li>
 <li><a href="https://cesare-montresor.github.io/idealNN/hierarchy.html">Classes</a></li>
 <li><a href="https://cesare-montresor.github.io/idealNN/files.html">Files</a></li>
</ul>


*/


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

    /// Default type for weak pointers
    template<typename T>
    using weak_ptr = std::weak_ptr<T>;
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
    using TensorDataRef = shared_ptr<TensorData>;
    /// Default type for TensorData arrays
    using TensorDataArray = vector<TensorDataRef> ;
    /// Default type for pointers to arrays of TensorData
    using TensorDataArrayRef = shared_ptr<TensorDataArray> ;

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
    /// Default type for weak Layer pointers, used to avoid circular ownership between Tensor and Layer
    using LayerWeakRef = weak_ptr<Layer>;
    /// Default type for array of Layers
    using LayerArray = vector<LayerRef>;
    /// Default type for pointers to array of Layers
    using LayerArrayRef = shared_ptr<LayerArray>;


}



#endif //IDEALNN_COMMON_H
