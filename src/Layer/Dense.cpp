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

        weights->data->setRandom();
        bias->data->setRandom();ù

        addParam(weight)
        addParam(bias)
    }

//Static
    TensorArrayRef Dense::forwardBatch(TensorArrayRef xs) {
        auto bs = xs->size();
        this->xs = xs;
        activations->clear();
        for(int i=0; i<bs; i++){
            auto x = xs->at(i);
            auto output = this->forward(x,i);
            activations->push_back(output);
        }
        return activations;
    }

    TensorRef Dense::forward(TensorRef x, ArrayIndex i) {
        auto result = ( (*weights->data) * (*x->data) + (*bias->data));
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
            output->extendOperations(x, shared_from_this());
        }
        return output;
    }


    void Dense::backward(TensorRef deltas, ArrayIndex i) {
        (*bias->gradients) = (*bias->gradients) + (*deltas->data);
        (*weights->gradients) = (*weights->gradients) + ( (*xs->at(i)->data ) * (*deltas->data) );
    }

    TensorArrayRef Dense::parameters() {
        auto params = Utils::MakeTensorArray();
        params->push_back(weights);
        params->push_back(bias);
        return params;
    }
}