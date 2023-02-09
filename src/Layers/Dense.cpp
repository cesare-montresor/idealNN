//
// Created by cesare on 08/02/23.
//

#include "Dense.h"
#include "../Tensor.h"


Dense::Dense(int in, int out){
    this->in = in;
    this->out = out;
}


DenseRef Dense::build(int in, int out) {
    return std::make_unique<Dense>(in, out);
}

Tensor Dense::forward(const Tensor& input)
{

}

void Dense::backward(){

}