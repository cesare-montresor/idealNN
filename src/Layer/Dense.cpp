//
// Created by cesare on 08/02/23.
//

#include "Dense.h"
#include "../Tensor/Tensor.h"
#include "../Utils.h"

//Constructors
Dense::Dense(int in, int out){
    this->in = in;
    this->out = out;

    weights = Utils::MakeMatrix(in,out);
    bias = Utils::MakeMatrix(out,1);
    activations = Utils::MakeMatrix(in, out);
    gradients = Utils::MakeMatrix(in, out);

    weights->setRandom();
    bias->setRandom();
    activations->setZero();
    gradients->setZero();
}

//Static
DenseRef Dense::MakeDense(int in, int out) {
    return make_unique<Dense>(in, out);
}


Tensor Dense::forward(const Tensor& input){

}

MatrixRef Dense::forward(const MatrixRef input){
    (*activations) = ( (*input) * (*weights) ) + (*bias);
    return (*activations);
}

void Dense::backward(){

}