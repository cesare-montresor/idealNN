//
// Created by cesare on 22/02/23.
//

#include <Activation/RELUActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    RELUActivationRef RELUActivation::MakeRELUActivation() { return std::make_shared<RELUActivation>(); }

    TensorRef RELUActivation::forward(TensorRef x, ArrayIndex i) {
        auto result = x->data->array().max(0).matrix();
        auto output = Tensor::MakeTensor(result);

        output->operation = shared_from_this();
        return output;
    }

    void RELUActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);

        auto x_rows = x->data->rows();
        auto x_cols = x->data->cols();

        auto dx_rows = dx->data->rows();
        auto dx_cols = dx->data->cols();


        std::cout << "[RELU] \t DIM X  [ " << std::endl << x_rows << " ; "<< x_cols << " ]" << std::endl << std::flush;
        std::cout << "[RELU] \t DIM DX [ " << std::endl << dx_rows << " ; "<< dx_cols << " ]" << std::endl << std::flush;


        auto zeros = Utils::MakeMatrix(dx->data->rows(), dx->data->cols())->setZero();
        auto relu_dx = (( x->data->array() <= 0 ).select(zeros.array(), dx->data->array() )).matrix();

        auto rdx_rows = relu_dx.rows();
        auto rdx_cols = relu_dx.cols();

        std::cout << "[RELU] \t DIM DX [ " << std::endl << rdx_rows << " ; "<< rdx_cols << " ]" << std::endl << std::flush;

        std::cout << "[GRADS] \t"<<i<<" RELU input DX" << std::endl << dx->data->array() << std::endl << std::flush;
        if(x->operation){
            auto next_dx = Tensor::MakeTensor( relu_dx.matrix() );
            std::cout << "[GRADS] \t"<<i<<" RELU (final)" << std::endl << next_dx->data->array() << std::endl << std::flush;
            x->operation->backward(next_dx,i);
        }
    }


} // IdealNN