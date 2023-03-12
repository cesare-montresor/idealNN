//
// Created by cesare on 15/02/23.
//

#include <Loss/CrossEntropyLoss.h>
#include <Utils.h>
#include <Layer/LinearLayer.h>

namespace IdealNN {
    CrossEntropyLossRef CrossEntropyLoss::MakeCrossEntropyLoss() { return std::make_shared<CrossEntropyLoss>(); }

    //https://www.youtube.com/watch?v=znqbtL0fRA0&t=2132s
    ScalarValue CrossEntropyLoss::loss(TensorArrayRef ys_hat, TensorArrayRef ys ){
        auto bs = Utils::getSize(ys_hat);
        this->ys_hat = ys_hat;
        deltas = Tensor::MakeTensorArray(bs);
        ScalarValue loss = 0;
        for(ArraySize i = 0 ; i<bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto log_y_hat = (y_hat->data->array() ).log();
            auto y_error = (y->data->array() * log_y_hat ).matrix();
            loss += y_error.array().sum();

            auto delta = Tensor::MakeTensor(((y->data->array() / y_hat->data->array() * -1 ).matrix()) );
            deltas->at(i) = delta;
        }
        return -loss / ScalarValue(bs);
    }

    void CrossEntropyLoss::backward(){
        auto bs = Utils::getSize( deltas );
        for(ArraySize i = 0; i<bs; i++) {
            auto delta = deltas->at(i);
            //std::cout << "[GRADS] \t"<<i<<" Cross Entropy loss \t" << delta->data->array() << std::endl << std::flush;
            auto y_hat = ys_hat->at(i);
            if(y_hat->operation) {
                y_hat->operation->backward(delta,i);
            }
        }
    }
}
