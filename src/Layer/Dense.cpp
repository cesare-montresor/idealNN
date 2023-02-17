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
    TensorArrayRef Dense::forward(TensorArrayRef batch) {
        activations->clear();
        for(int i=0; i<batch->size(); i++){
            auto input = (*batch)[i];
            auto result = ( (*weights->data) * (*input->data) + (*bias->data));
            auto output = Tensor::MakeTensor(result);
            output->extendOperations(input, shared_ptr<Dense>(this)) ;
            activations->push_back(output);
        }
        return activations;
    }



    void Dense::backward() {

    }

}