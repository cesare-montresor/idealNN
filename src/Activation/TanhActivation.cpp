//
// Created by cesare on 22/02/23.
//


#include <Activation/TanhActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {

    TanhActivationRef TanhActivation::MakeTanhActivation() { return std::make_shared<TanhActivation>(); }

    TensorRef TanhActivation::forward(TensorRef x) {
        auto result =  x->data->array().tanh().matrix();
        auto output = Tensor::MakeTensor(result);

        output->operation = weak_from_this();
        return output;
    }

    void TanhActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = inputs->at(i);
        auto output = outputs->at(i);
        auto tanh_2 = output->data->array().pow(2);

        auto tanh_dx = (1-tanh_2).matrix() ;
        //std::cout << "[GRADS] \t"<<i<<" Sigmoid (partial)" << std::endl << sigma_dx.array() << std::endl << std::flush;
        auto operation = x->operation.lock();
        if(operation){
            auto next_dx = Tensor::MakeTensor( (tanh_dx.array() * dx->data->array() ).matrix() );
            //std::cout << "[GRADS] \t"<<i<<" Sigmoid (final)" << std::endl << next_dx->data->array() << std::endl << std::flush;
            operation->backward(next_dx,i);
        }
    }

} // IdealNN