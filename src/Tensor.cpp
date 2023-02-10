//
// Created by cesare on 08/02/23.
//

#include "Tensor.h"

using Eigen::MatrixXd;
using Eigen::EigenBase;

Tensor::Tensor(DimList dims){

}

Tensor::Tensor(MatrixXd data){
    this->data = data;
    this->dims =data.
}

