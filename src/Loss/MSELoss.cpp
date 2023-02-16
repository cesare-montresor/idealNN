//
// Created by cesare on 15/02/23.
//

#include "MSELoss.h"

namespace IdealNN {
    Scalar MSELoss::loss(TensorArrayRef ys, TensorArrayRef ys_hat ){
        auto errors = ScalarArray(ys->size() );
        Scalar loss = 0;
        for(int i = 0 ; i< ys->size(); i++){
            auto y = (*ys)[i];
            auto y_hat = (*ys_hat)[i];
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            errors[i] = error;
            loss += error * error;
        }
        return loss/Scalar(ys->size());
    }
}
