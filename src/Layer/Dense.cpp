//
// Created by cesare on 08/02/23.
//

#include "Dense.h"
#include "../Tensor/Tensor.h"
#include "../Utils.h"

namespace IdealNN {
//Constructors
    Dense::Dense(int in, int out) {
        this->in = in;
        this->out = out;

        weights = Tensor::MakeTensor(in, out);
        bias = Tensor::MakeTensor(out, 1);
        activations = Tensor::MakeTensor(in, out);
        gradients = Tensor::MakeTensor(in, out);

        weights->data->setRandom();
        bias->data->setRandom();
        activations->data->setZero();
        gradients->data->setZero();
    }

//Static
    DenseRef Dense::MakeDense(int in, int out) {
        return make_unique<Dense>(in, out);
    }

    TensorRef Dense::forward(TensorRef input) {
        auto result = ((*input->data) * (*weights->data)) + (*bias->data);
        activations = Tensor::MakeTensor( result );
        return activations;
    }


    TensorRef Dense::forward(Tensor const &input) {
        auto tensor = Tensor::MakeTensor(input);
        return forward( tensor );
    }

    void Dense::backward() {

    }

}