//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_ACTIVATION_H
#define IDEALNN_ACTIVATION_H

#include <Layer/Layer.h>


namespace IdealNN {
    /// Represents a general purpose interface for Activation layers/functions
    struct Activation: public Layer {
        /// Activation layers have no params to train, always return an empty vector.
        TensorArrayRef parameters() final;
    };
}

#endif //IDEALNN_ACTIVATION_H
