//
// Created by cesare on 22/02/23.
//


#include "SoftmaxActivation.h"
#include "../Utils.h"
#include "../Tensor/Tensor.h"


namespace IdealNN {

    SoftmaxActivation::SoftmaxActivation(){
        activations = Utils::MakeTensorArray();

    }

    TensorArrayRef SoftmaxActivation::forwardBatch(TensorArrayRef xs) {
        auto bs = xs->size();
        this->xs = xs;
        this->xs_exp = Utils::MakeMatrixArray(bs);
        this->xs_exp_sum = Utils::MakeScalarValueArray(bs);
        activations->clear();
        for(int i=0; i<bs; i++){
            auto x = xs->at(i);
            auto output = this->forward(x,i);
            activations->push_back(output);
        }
        return activations;
    }

    TensorRef SoftmaxActivation::forward(TensorRef x, ArrayIndex i) {
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

    void SoftmaxActivation::backward(TensorRef dx, ArrayIndex i) {
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

    TensorArrayRef SoftmaxActivation::parameters() {
        return Utils::MakeTensorArray();
    }
} // IdealNN