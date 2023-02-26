//
// Created by cesare on 22/02/23.
//


#include "SigmoidActivation.h"
#include "../Utils.h"
#include "../Tensor/Tensor.h"


namespace IdealNN {

    SigmoidActivation::SigmoidActivation(){
        activations = Utils::MakeTensorArray();
    }

    TensorArrayRef SigmoidActivation::forwardBatch(TensorArrayRef xs) {
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

    TensorRef SigmoidActivation::forward(TensorRef x, ArrayIndex i) {
        // Better numerical stability (?)
        //auto x_stable = x->data->array() - x->data->array().max(x->data->array());
        auto x_exp = x->data->array().exp();
        auto x_exp_sum = x_exp.sum();
        auto result = Matrix((x_exp / x_exp_sum).matrix());
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
            output->extendOperations(x, shared_from_this());
        }
        return output;
    }

    void SigmoidActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto sigma_x = activations->at(i)->data->array();
        auto sigma_dx = Matrix( (( 1 - sigma_x ) * sigma_x).matrix() );
        auto ops_num = x->operations->size();
        if(ops_num==0) return;

        auto prevLayer = x->operations->back();
        x->operations->pop_back();
        auto next_dx = Tensor::MakeTensor( sigma_dx );

        prevLayer->backward( next_dx, i );
    }

    TensorArrayRef SigmoidActivation::parameters() {
        return Utils::MakeTensorArray();
    }
} // IdealNN