//
// Created by cesare on 08/02/23.
//

#include <Layer/Dense.h>
#include <Tensor/Tensor.h>
#include <Utils.h>

namespace IdealNN {
//Constructors
    Dense::Dense(int in, int out):Layer() {
        this->in = in;
        this->out = out;

        weights = Tensor::MakeTensor(out, in);
        bias = Tensor::MakeTensor(out, 1);

        weights->data->setRandom();
        bias->data->setRandom();
        weights->zero_grad();
        bias->zero_grad();
    }

//Static


    TensorRef Dense::forward(TensorRef x, ArrayIndex i) {
        auto result = ( (*weights->data) * (*x->data) + (*bias->data));
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
            output->extendOperations(x, shared_from_this());
        }
        return output;
    }


    void Dense::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto bias_dx = (*dx->data);
        auto weights_dx = bias_dx * (x->data->transpose()) ;
        (*bias->gradients) += bias_dx;
        (*weights->gradients) += weights_dx;

        auto ops_num = x->operations->size();
        if(ops_num==0) return;

        auto prevLayer = x->operations->back();
        x->operations->pop_back();
        auto next_dx = Tensor::MakeTensor( weights_dx.transpose() );

        prevLayer->backward( next_dx, i );
    }

    TensorArrayRef Dense::parameters() {
        auto params = Utils::MakeTensorArray();
        params->push_back(weights);
        params->push_back(bias);
        return params;
    }
}