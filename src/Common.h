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
    using DimList = vector<int>;
    using LayerList = vector<int>;


    typedef float Scalar;
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


}



#endif //IDEALNN_COMMON_H
