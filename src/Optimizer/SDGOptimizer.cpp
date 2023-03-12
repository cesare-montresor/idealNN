//
// Created by cesare on 20/02/23.
//

#include <Optimizer/SDGOptimizer.h>
#include <Utils.h>
#include <Layer/Layer.h>

namespace IdealNN{

    SDGOptimizerRef SDGOptimizer::MakeSDGOptimizer(const LayerArrayRef& layers, ScalarValue learning_rate) { return std::make_shared<SDGOptimizer>(layers, learning_rate); }


    SDGOptimizer::SDGOptimizer(const LayerArrayRef& layers, ScalarValue learning_rate) {
        this->learning_rate = learning_rate;
        parameters = Tensor::MakeTensorArray();
        auto ls = Utils::getSize(layers);
        for(ArraySize i = 0; i< ls; i++){
            auto params = layers->at(i)->parameters();
            auto ps = Utils::getSize(params);
            for(ArraySize j=0; j<ps; j++){
                parameters->push_back( params->at(j) );
            }
        }
    }

    /*
    SDGOptimizer::SDGOptimizer(const TensorArrayRef& params, ScalarValue learning_rate){
        this->learning_rate = learning_rate;
        this->parameters = Tensor::MakeTensorArray();
        auto ps = Utils::getSize(params);
        for(ArraySize j=0; j<ps; j++){
            this->parameters->push_back( params->at(j) );
        }
    }
    */


    void SDGOptimizer::step(){
        //TODO: OpenMP parallel
        auto ps = Utils::getSize(parameters);
        for(ArraySize i=0; i< ps; i++){
            auto param = parameters->at(i);
            auto grads = param->gradients;
            (*param->data) -= (*grads) * learning_rate;
        }
    }

    void SDGOptimizer::zero_grad(){
        //TODO: OpenMP parallel
        auto ps = Utils::getSize(parameters);
        for(ArraySize i=0; i< ps; i++){
            auto param = parameters->at(i);
            param->zero_grad();
        }
    }


}