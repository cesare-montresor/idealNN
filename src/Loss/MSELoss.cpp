//
// Created by cesare on 15/02/23.
//

#include "MSELoss.h"
#include "../Utils.h"

namespace IdealNN {


    ScalarValue MSELoss::loss(TensorArrayRef ys, TensorArrayRef ys_hat ){
        auto bs = ys->size();
        this->ys = ys;
        deltas = Utils::MakeScalarArray(bs);
        ScalarValue loss = 0;
        for(int i = 0 ; i<bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto y_error = (*y->data) - (*y_hat->data);
            deltas->at(i) = Scalar::MakeScalar(y_error);
            if(deltas->at(i)->use_grads){
                deltas->at(i)->inheritOperations(y_hat);
            }
            loss += pow(y_error.value(), 2);
        }
        loss = loss / ScalarValue(bs) ;
        return loss;
    }

    void MSELoss::backward(){
        auto bs = deltas->size();
        for(int i = 0; i<bs; i++) {
            auto delta = deltas->at(i);
            auto ops_num = delta->operations->size();
            if(ops_num>0) {
                auto prevLayer = delta->operations->back();
                delta->operations->pop_back();

            }
        }
    }
}
