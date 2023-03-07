//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#include <Layer/Layer.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {

    //static array
    TensorArrayRef Layer::forwardBatch(TensorArrayRef xs) {
        this->xs = xs;
        auto bs = Utils::getSize(xs);
        auto activations = Tensor::MakeTensorArray(bs);
        for(ArraySize i=0; i<bs; i++){
            auto x = xs->at(i);
            auto output = this->forward(x,i);
            activations->at(i) = output;
        }
        return activations;
    }

}
