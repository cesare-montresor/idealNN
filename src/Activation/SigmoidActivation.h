//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_SIGMOIDACTIVATION_H
#define IDEALNN_SIGMOIDACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {
    struct SigmoidActivation:  public Activation {
        TensorRef forward(TensorRef x, ArrayIndex i) override;
        void backward(TensorRef dx, ArrayIndex i) override;
    };

} // IdealNN

#endif //IDEALNN_SIGMOIDACTIVATION_H
