//
// Created by cesare on 15/02/23.
//

#include <Loss/MSELoss.h>
#include <Utils.h>
#include <Layer/Dense.h>

namespace IdealNN {

    // https://www.youtube.com/watch?v=d9AvALaC-5s
    ScalarValue MSELoss::loss( TensorArrayRef ys_hat, TensorArrayRef ys ){
        auto bs = Utils::getSize(ys);
        this->ys_hat = ys_hat;
        deltas = Tensor::MakeTensorArray(bs);
        ScalarValue loss = 0;
        for(ArraySize i = 0 ; i<bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto y_error = (*y_hat->data) - (*y->data); // final derivative step ->  (y_hat - y)
            deltas->at(i) = Tensor::MakeTensor(y_error);
            loss += ((y_error.array().pow(2)/2).sum() ) / ((ScalarValue)(y_error.array().size()));
        }
        loss = loss / ScalarValue(bs) ;
        return loss;
    }

    void MSELoss::backward(){
        auto bs = Utils::getSize(ys_hat);
        for(ArraySize i = 0; i<bs; i++) {
            auto delta = deltas->at(i);
            auto y_hat = ys_hat->at(i);
            //std::cout << "[GRADS] \t"<<i<<" MSELoss (final) " << std::endl << delta->data->transpose().array() << std::endl << std::flush;
            if(y_hat->operation) {
                y_hat->operation->backward(delta, i);
            }
        }
    }
}
