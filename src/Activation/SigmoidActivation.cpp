//
// Created by cesare on 22/02/23.
//


#include <Activation/SigmoidActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    SigmoidActivationRef SigmoidActivation::MakeSigmoidActivation() { return std::make_shared<SigmoidActivation>(); }

    TensorRef SigmoidActivation::forward(TensorRef x) {
        auto sigma_exp = ((*x->data) * -1).array().exp();
        auto result = ( 1 / ( 1 + sigma_exp)).matrix();
        auto output = Tensor::MakeTensor(result);

        output->operation = weak_from_this();
        return output;
    }

    void SigmoidActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = inputs->at(i);
        auto output = outputs->at(i);
        auto sigma_dx = (( 1 - output->data->array() ) * output->data->array()).matrix() ;

        //std::cout << "[GRADS] \t"<<i<<" Sigmoid (partial)" << std::endl << sigma_dx.array() << std::endl << std::flush;
        auto operation = x->operation.lock();
        if(operation){
            auto next_dx = Tensor::MakeTensor( (sigma_dx.array() * dx->data->array() ).matrix() );
            //std::cout << "[GRADS] \t"<<i<<" Sigmoid (final)" << std::endl << next_dx->data->array() << std::endl << std::flush;
            operation->backward(next_dx,i);
        }
    }

} // IdealNN