//
// Created by cesare on 15/02/23.
//

#include <Loss/CrossEntropyLoss.h>
#include <Utils.h>
#include <Layer/Dense.h>

namespace IdealNN {


    ScalarValue CrossEntropyLoss::loss(TensorArrayRef ys_hat, TensorArrayRef ys ){
        auto bs = Utils::toArraySize(ys_hat->size());
        this->ys_hat = ys_hat;
        deltas = Utils::MakeTensorArray(bs);
        ScalarValue loss = 0;
        for(ArraySize i = 0 ; i<bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto log_y_hat = (y_hat->data->array() ).log();
            //std::cout << "log_y: " << y_hat->data->array() << std::endl << std::flush;
            //std::cout << "log_y_hat: " << log_y_hat.array() << std::endl << std::flush;
            auto y_error = Matrix( (y->data->array() * log_y_hat ).matrix() );
            deltas->at(i) = Tensor::MakeTensor(y_error);
            if(deltas->at(i)->use_grads){
                deltas->at(i)->inheritOperations(y_hat);
            }
            loss += y_error.array().sum();
        }
        return -loss / ScalarValue(bs);
    }

    void CrossEntropyLoss::backward(){
        auto bs = Utils::toArraySize( deltas->size() );
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
