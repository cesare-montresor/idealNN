//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_RELUACTIVATION_H
#define IDEALNN_RELUACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {

    struct RELUActivation:  public Activation {
        TensorRef forward(TensorRef x, ArrayIndex i) override;
        void backward(TensorRef dx, ArrayIndex i) override;
    };

} // IdealNN

#endif //IDEALNN_RELUACTIVATION_H
