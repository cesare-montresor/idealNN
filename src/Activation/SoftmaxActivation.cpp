//
// Created by cesare on 22/02/23.
//


#include <Activation/SoftmaxActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    SoftmaxActivationRef SoftmaxActivation::MakeSoftmaxActivation() { return std::make_shared<SoftmaxActivation>(); }

    TensorRef SoftmaxActivation::forward(TensorRef x, ArrayIndex i) {
        auto x_exp = x->data->array().exp();
        auto x_exp_sum = x_exp.sum();
        auto result = (x_exp / x_exp_sum).matrix();
        //std::cout << "[FORWARD] \t" << i << " Softmax X " << x->data->array() << std::endl << std::flush;
        //std::cout << "[FORWARD] \t" << i << " Softmax A " << result.array() << std::endl << std::flush;

        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
        }
        return output;
    }

    void SoftmaxActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto x_cnt = x->data->cols();
        auto x_exp = x->data->array().exp();
        auto x_exp_sum = x_exp.sum();
        auto softmax = (x_exp / x_exp_sum).matrix();

        auto softmax_dx = Utils::MakeMatrix( x->data->rows(),x->data->cols());
        softmax_dx->setZero();
        for(int j = 0; j < x_cnt; j++ ){
            softmax_dx->coeffRef(j) = softmax.coeff(j) * (1 - softmax.coeff(j) );;
        }
        //std::cout << "[GRADS] \t"<<i<<" Softmax (local)" << std::endl << softmax_dx->array() << std::endl << std::flush;
        if(x->operation){
            auto next_dx = Tensor::MakeTensor( (softmax_dx->array() * dx->data->array()).matrix()  );
            //std::cout << "[GRADS] \t"<<i<<" Softmax (final)" << std::endl << next_dx->data->array() << std::endl << std::flush;
            x->operation->backward(next_dx,i);
        }
    }


} // IdealNN


/*
for(int k = 0; k < x_cnt; k++ ){
    if(j == k){
        grad += softmax.coeff(j) * (1 - softmax.coeff(j) );
    }else{
        grad += -( softmax.coeff(j) * softmax.coeff(k) );
    }
}
*/


