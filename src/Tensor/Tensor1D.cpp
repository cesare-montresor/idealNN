//
// Created by cesare on 08/02/23.
//

#include "Tensor1D.h"

using Eigen::MatrixXd;
using Eigen::EigenBase;

Tensor1D::Tensor1D(){
    this->dims = dims;
    this->data = MatrixXd(dims.at(0), dims.at(0));

}

Tensor1D::Tensor1D(MatrixXd data){
    this->data = data;

}

