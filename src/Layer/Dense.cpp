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

        weights = Tensor::MakeTensor(out, in);
        bias = Tensor::MakeTensor(out, 1);
        activations = Utils::MakeTensorArray();
        gradients = Tensor::MakeTensor(in, out);

        weights->data->setRandom();
        bias->data->setRandom();
        gradients->data->setZero();
    }

//Static
    TensorArrayRef Dense::forward(TensorArrayRef xs) {
        auto bs = xs->size();
        this->xs = xs;
        activations->clear();
        for(int i=0; i<bs; i++){
            auto input = xs->at(i);
            auto result = ( (*weights->data) * (*input->data) + (*bias->data));
            auto output = Tensor::MakeTensor(result);
            if(input->use_grads) {
                output->operation = shared_from_this();
                output->extendOperations(input, shared_from_this());
            }
            activations->push_back(output);
        }
        return activations;
    }


    void Dense::backward(TensorArrayRef deltas) {

    }

}