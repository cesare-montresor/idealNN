//
// Created by cesare on 20/02/23.
//

#include "SDGOptimizer.h"
#include "../Utils.h"
#include "../Layer/Layer.h"

namespace IdealNN{

    SDGOptimizer::SDGOptimizer(const LayerArrayRef& layers, ScalarValue learning_rate){
        this->learning_rate = learning_rate;
        this->parameters = Utils::MakeTensorArray();
        for(int i = 0; i< layers->size(); i++){
            auto params = layers->at(i)->parameters();
            for(int j=0; j<params->size(); j++){
                this->parameters->push_back( params->at(j) );
            }
        }
    }

    SDGOptimizer::SDGOptimizer(const TensorArrayRef& params, ScalarValue learning_rate){
        this->learning_rate = learning_rate;
        this->parameters = Utils::MakeTensorArray();
        for(int j=0; j<params->size(); j++){
            this->parameters->push_back( params->at(j) );
        }
    }


    void SDGOptimizer::step(){
        //TODO: OpenMP parallel
        for(int i=0; i< this->parameters->size(); i++){
            auto param = this->parameters->at(i);
            auto grads = param->gradients;
            (*param->data) -= (*grads) * learning_rate;
        }
    }

    void SDGOptimizer::zero_grad(){
        //TODO: OpenMP parallel
        for(int i=0; i< this->parameters->size(); i++){
            auto param = this->parameters->at(i);
            param->zero_grad();
        }
    }


}