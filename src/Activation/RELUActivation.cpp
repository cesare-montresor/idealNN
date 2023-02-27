//
// Created by cesare on 22/02/23.
//

#include <Activation/RELUActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    RELUActivation::RELUActivation(){
        activations = Utils::MakeTensorArray();
    }

    TensorArrayRef RELUActivation::forwardBatch(TensorArrayRef xs) {
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

    TensorRef RELUActivation::forward(TensorRef x, ArrayIndex i) {
        auto result = x->data->array().max(0).matrix();
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
            output->extendOperations(x, shared_from_this());
        }
        return output;
    }

    void RELUActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto zeros = Utils::MakeMatrix(x->data->rows(), x->data->cols())->setZero();
        auto relu_dx_op  = ( x->data->array() <= 0 ).select(zeros.array(), dx->data->array());
        auto relu_dx = Matrix(relu_dx_op.matrix() );

        auto ops_num = x->operations->size();
        if(ops_num==0) return;

        auto prevLayer = x->operations->back();
        x->operations->pop_back();
        auto next_dx = Tensor::MakeTensor( relu_dx );

        prevLayer->backward( next_dx, i );
    }

    TensorArrayRef RELUActivation::parameters() {
        return Utils::MakeTensorArray();
    }



} // IdealNN