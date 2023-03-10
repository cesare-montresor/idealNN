//
// Created by cesare on 22/02/23.
//


#include <Activation/SoftmaxActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    SoftmaxActivationRef SoftmaxActivation::MakeSoftmaxActivation() { return std::make_shared<SoftmaxActivation>(); }

    TensorRef SoftmaxActivation::forward(TensorRef x) {
        auto x_stable = (x->data->array() - x->data->array().maxCoeff()).matrix();
        auto x_exp = x_stable.array().exp();
        auto x_exp_sum = x_exp.sum();
        auto result = (x_exp / x_exp_sum).matrix();
        //std::cout << "[FORWARD] \t" << i << " Softmax X " << x->data->array() << std::endl << std::flush;
        //std::cout << "[FORWARD] \t" << i << " Softmax A " << result.array() << std::endl << std::flush;

        auto output = Tensor::MakeTensor(result);
        output->operation = shared_from_this();
        return output;
    }

    void SoftmaxActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = inputs->at(i);
        auto output = outputs->at(i);

        //std::cout << "[GRADS] \t"<<i<<" Softmax (x) \t\t" << x->data->array() << std::endl << std::flush;
        //std::cout << "[GRADS] \t"<<i<<" Softmax (x_stable) \t\t" << x_stable.array() << std::endl << std::flush;
        auto softmax_dx = (output->data->array() * (1 - output->data->array())).matrix();
        //std::cout << "[GRADS] \t"<<i<<" Softmax (local) \t" << softmax_dx.array() << std::endl << std::flush;
        if(x->operation){
            auto next_dx = Tensor::MakeTensor( (softmax_dx.array() * dx->data->array()).matrix()  );
            //std::cout << "[GRADS] \t"<<i<<" Softmax (final) \t" << next_dx->data->array() << std::endl << std::flush;
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


