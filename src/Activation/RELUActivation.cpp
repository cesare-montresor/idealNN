//
// Created by cesare on 22/02/23.
//

#include <Activation/RELUActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {

    TensorRef RELUActivation::forward(TensorRef x, ArrayIndex i) {
        auto result = x->data->array().max(0).matrix();
        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
        }
        return output;
    }

    void RELUActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto zeros = Utils::MakeMatrix(x->data->rows(), x->data->cols())->setZero();
        auto relu_dx_op  = ( x->data->array() <= 0 ).select(zeros.array(), dx->data->array());
        auto relu_dx = Matrix(relu_dx_op.matrix() );

        //std::cout << "[GRADS] \t"<<i<<" RELU (partial)" << std::endl << relu_dx.array() << std::endl << std::flush;
        if(x->operation){
            auto next_dx = Tensor::MakeTensor( relu_dx );
            //std::cout << "[GRADS] \t"<<i<<" RELU (final)" << std::endl << next_dx->data->transpose().array() << std::endl << std::flush;
            x->operation->backward(next_dx,i);
        }
    }


} // IdealNN