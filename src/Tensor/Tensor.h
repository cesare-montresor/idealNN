//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H


#include <Eigen/Dense>
#include "../Common.h"

namespace IdealNN {


    struct Tensor {
        Eigen::MatrixXd data;
        DimList dims;

        Tensor(DimList dims);

        Tensor(Eigen::MatrixXd data);
    };
}

#endif //IDEALNN_TENSOR_H
