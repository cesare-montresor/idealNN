//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_COMMON_H
#define IDEALNN_COMMON_H

#include <vector>
#include <memory>
#include <Eigen/Eigen>


using namespace std;

using DimList = vector<int>;
using LayerList = vector<int>;


typedef float Scalar;
typedef Eigen::RowVectorXf VectorRow;
typedef Eigen::VectorXf VectorCol;
typedef Eigen::MatrixXf Matrix;

typedef vector<Scalar> ScalarArray;
typedef vector<VectorRow*> VectorRowArray;
typedef vector<VectorCol*> VectorColArray;
typedef vector<Matrix*> MatrixArray;

typedef unsigned long int ArrayIndex;
typedef unsigned long int ArraySize;

using VectorRowRef = unique_ptr<VectorRow>;
using VectorColRef = unique_ptr<VectorCol>;
using MatrixRef = unique_ptr<Matrix>;




#endif //IDEALNN_COMMON_H
