//
// Created by cesare on 22/02/23.
//


#include <Activation/SigmoidActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {

    TensorRef SigmoidActivation::forward(TensorRef x, ArrayIndex i) {
        auto sigma_exp = ((*x->data) * -1).array().exp();
        auto result = ( 1 / ( 1 + sigma_exp)).matrix();
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
        }
        return output;
    }

    void SigmoidActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto sigma_exp = ((*x->data) * -1).array().exp();
        auto sigma_x = ( 1 / ( 1 + sigma_exp));

        auto sigma_dx = Matrix( (( 1 - sigma_x ) * sigma_x).matrix() );
        if(x->operation){
            auto next_dx = Tensor::MakeTensor( sigma_dx );
            x->operation->backward(next_dx,i);
        }
    }

} // IdealNN