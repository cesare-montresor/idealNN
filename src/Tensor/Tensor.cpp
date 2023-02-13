//
// Created by cesare on 08/02/23.
//

#include "Tensor.h"

using Eigen::MatrixXd;
using Eigen::EigenBase;

Tensor::Tensor(DimList dims){
    int len = dims.size();
    
    while(dims.size() < 5){
        dims.push_back(-1);
    }
    this->dims = dims;
    this->data = MatrixXd(dims.at(0), dims.at(0));

}

Tensor::Tensor(MatrixXd data){
    this->data = data;
}

