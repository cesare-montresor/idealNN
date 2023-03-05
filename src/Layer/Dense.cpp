//
// Created by cesare on 08/02/23.
//

#include <Layer/Dense.h>
#include <Tensor/Tensor.h>
#include <Utils.h>

namespace IdealNN {
//Constructors
    Dense::Dense(ArraySize  in, ArraySize  out):Layer() {
        this->in = in;
        this->out = out;

        weights = Tensor::MakeTensor(out, in);
        bias = Tensor::MakeTensor(1, out);


        weights->data->setRandom();
        bias->data->setZero();
        weights->zero_grad();
        bias->zero_grad();
    }

    TensorRef Dense::forward(TensorRef x, ArrayIndex i) {
        auto result = ( (*x->data) * weights->data->transpose() + (*bias->data) );
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
        }
        return output;
    }

    void Dense::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i); // xs is the batch, x is the single input

        //std::cout << "[GRADS] \t" << i << " Dense (local) " << std::endl << x->data->array() << std::endl << std::flush;
        auto bias_dx = (*dx->data);
        auto weights_dx = dx->data->transpose() * (*x->data);
        //std::cout << "[GRADS] \t" << i << " Dense (final) " << std::endl << weights_dx.array() << std::endl << std::flush;
        (*bias->gradients) += bias_dx;
        (*weights->gradients) += weights_dx;

        if(x->operation) {
            auto next_dx = Tensor::MakeTensor( weights_dx );
            x->operation->backward(next_dx, i);
        }
    }

    TensorArrayRef Dense::parameters() {
        auto params = Utils::MakeTensorArray();
        params->push_back(weights);
        params->push_back(bias);
        return params;
    }
}