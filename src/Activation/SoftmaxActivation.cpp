//
// Created by cesare on 22/02/23.
//


#include <Activation/SoftmaxActivation.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {

    TensorRef SoftmaxActivation::forward(TensorRef x, ArrayIndex i) {
        auto x_exp = x->data->array().exp();
        auto x_exp_sum = x_exp.sum();
        auto result = Matrix((x_exp / x_exp_sum).matrix());

        auto output = Tensor::MakeTensor(result);
        if(x->use_grads) {
            output->operation = shared_from_this();
        }
        return output;
    }

    void SoftmaxActivation::backward(TensorRef dx, ArrayIndex i) {
        auto x = xs->at(i);
        auto x_num_row = x->data->rows();
        auto x_num_col = x->data->cols();
        auto x_cnt = x_num_row;
        auto x_exp = x->data->array().exp();
        auto x_exp_sum = x_exp.sum();
        auto softmax = Matrix((x_exp / x_exp_sum).matrix());

        auto softmax_dx = Utils::MakeMatrix(x_num_row, x_num_col);
        softmax_dx->setZero();
        for(int i = 0; i < x_cnt; i++ ){
            for(int j = 0; j < x_cnt; j++ ){
                if(i == j){
                    softmax_dx->coeffRef(i) += softmax.coeff(i) * (1 - softmax.coeff(i) );
                }else{
                    softmax_dx->coeffRef(i) += -( softmax.coeff(i) * softmax.coeff(j) );
                }
            }
        }

        if(x->operation){
            auto next_dx = Tensor::MakeTensor( Matrix((softmax_dx->array()) * (dx->data->array())) );
            x->operation->backward(next_dx,i);
        }
    }


} // IdealNN