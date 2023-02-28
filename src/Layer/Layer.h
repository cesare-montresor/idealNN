//#include <Eigen/Dense>

//
// Created by Cesare on 25/10/2022.
//

#ifndef IDEALNN_LAYER_H
#define IDEALNN_LAYER_H

#include <vector>
#include <memory>
#include <Common.h>



namespace IdealNN {

    struct ILayer: public std::enable_shared_from_this<Layer> {

        //static array
        virtual TensorArrayRef forwardBatch(TensorArrayRef xs) = 0 ;
        virtual TensorRef forward(TensorRef x, ArrayIndex i) = 0 ;
        virtual void backward(TensorRef dx, ArrayIndex i) = 0;
        virtual TensorArrayRef parameters() = 0;
    };

    struct Layer: public ILayer {
    protected:
        TensorArrayRef xs;

    public:
        //static array
        TensorArrayRef forwardBatch(TensorArrayRef xs) override;
    };


}



#endif //IDEALNN_LAYER_H
