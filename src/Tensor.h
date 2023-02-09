//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H

#include <Eigen/Dense>
#include <vector>

using TensorDims = std::vector<int>;

struct Tensor {
    TensorDims dims;
    Tensor(TensorDims dims);
};


#endif //IDEALNN_TENSOR_H
