//
// Created by cesare on 22/02/23.
//

#include <Activation/RELUActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    RELUActivationRef RELUActivation::MakeRELUActivation() { return std::make_shared<RELUActivation>(); }

    TensorRef RELUActivation::forward(TensorRef x) {
        auto result = x->data->array().max(0).matrix();
        auto output = Tensor::MakeTensor(result);

        output->operation = shared_from_this();
        return output;
    }

    void RELUActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = inputs->at(i);

        auto zeros = Utils::MakeMatrix(dx->data->rows(), dx->data->cols())->setZero();
        auto relu_dx = (( x->data->array() <= 0 ).select(zeros.array(), dx->data->array() )).matrix();

        //std::cout << "[GRADS] \t"<<i<<" RELU input DX" << std::endl << dx->data->array() << std::endl << std::flush;
        if(x->operation){
            auto next_dx = Tensor::MakeTensor( relu_dx.matrix() );
            //std::cout << "[GRADS] \t"<<i<<" RELU (final)" << std::endl << next_dx->data->array() << std::endl << std::flush;
            x->operation->backward(next_dx,i);
        }
    }


} // IdealNN