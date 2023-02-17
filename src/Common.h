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

    typedef float Scalar;
    const Scalar ScalarDelta = 0.00001;

    //typedef Eigen::RowVectorXf VectorRow;
    //typedef Eigen::VectorXf VectorCol;
    typedef Eigen::MatrixXf Matrix;

    //using VectorRowRef = shared_ptr<VectorRow>;
    //using VectorColRef = shared_ptr<VectorCol>;
    using MatrixRef = shared_ptr<Matrix>;

    typedef vector<Scalar> ScalarArray;
    //typedef vector<VectorRowRef> VectorRowArray;
    //typedef vector<VectorColRef> VectorColArray;
    typedef vector<MatrixRef> MatrixArray;

    typedef unsigned long int ArrayIndex;
    typedef unsigned long int ArraySize;
    using DimList = vector<ArraySize>;


    // resolve circular deps Tensor <-> Layer
    struct Tensor;
    using TensorRef = shared_ptr<Tensor>;
    typedef vector<TensorRef> TensorArray;
    typedef shared_ptr<TensorArray> TensorArrayRef;


    struct Layer;
    using LayerRef = shared_ptr<Layer>;
    typedef vector<LayerRef> LayerArray;
    typedef shared_ptr<LayerArray> LayerArrayRef;

    struct Dense;
    using DenseRef = shared_ptr<Dense>;
}



#endif //IDEALNN_COMMON_H
