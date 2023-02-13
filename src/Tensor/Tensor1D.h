//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR1D_H
#define IDEALNN_TENSOR1D_H


#include "../Common.h"
#include <Eigen/Dense>

namespace IdealNN {
    struct Tensor1D {
        Eigen::MatrixXd data;
        DimList dims;

        Tensor1D();

        Tensor1D(Eigen::MatrixXd data);
    };
}

#endif //IDEALNN_TENSOR1D_H
