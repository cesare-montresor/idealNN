//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#include <Layer/Layer.h>
#include <Utils.h>
#include <Tensor/Tensor.h>


namespace IdealNN {

    //static array
    TensorArrayRef Layer::forwardBatch(const TensorArrayRef &xs) {
        this->inputs = xs;
        auto bs = Utils::getSize(xs);
        outputs = Tensor::MakeTensorArray(bs);
        for(ArraySize i=0; i<bs; i++){
            auto x = xs->at(i);
            auto output = this->forward(x);
            outputs->at(i) = output;
        }
        return outputs;
    }

}
