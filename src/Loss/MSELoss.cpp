//
// Created by cesare on 15/02/23.
//

#include <Loss/MSELoss.h>
#include <Utils.h>
#include <Layer/Dense.h>

namespace IdealNN {


    ScalarValue MSELoss::loss( TensorArrayRef ys_hat, TensorArrayRef ys ){
        auto bs = Utils::toArraySize(ys->size());
        this->ys_hat = ys_hat;
        deltas = Utils::MakeTensorArray(bs);
        ScalarValue loss = 0;
        for(ArraySize i = 0 ; i<bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto y_error = (*y->data) - (*y_hat->data);
            deltas->at(i) = Tensor::MakeTensor(y_error);
            if(deltas->at(i)->use_grads){
                deltas->at(i)->inheritOperations(y_hat);
            }
            loss += pow(y_error.value(), 2);
        }
        loss = loss / ScalarValue(bs) ;
        return loss;
    }

    void MSELoss::backward(){
        auto bs = Utils::toArraySize(deltas->size());
        for(ArraySize i = 0; i<bs; i++) {
            auto delta = deltas->at(i);
            auto ops_num = delta->operations->size();
            if(ops_num==0) continue;

            auto prevLayer = delta->operations->back();
            delta->operations->pop_back();
            prevLayer->backward(delta,i);
        }
    }
}
