//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_ACTIVATION_H
#define IDEALNN_ACTIVATION_H

#include <Layer/Layer.h>


namespace IdealNN {

    struct Activation: public Layer {
        TensorArrayRef parameters() override;
    };
}

#endif //IDEALNN_ACTIVATION_H
