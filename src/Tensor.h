//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H

#include <Eigen/Dense>
#include <list>

struct Tensor {
    std::list<int> dims;
    Tensor(std::list<int> dims);
};


#endif //IDEALNN_TENSOR_H
