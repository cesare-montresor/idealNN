//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_SIGMOIDACTIVATION_H
#define IDEALNN_SIGMOIDACTIVATION_H

#include "Activation.h"

namespace IdealNN {
    struct SigmoidActivation:  public Activation {
        TensorArrayRef xs;
        TensorArrayRef activations;

        SigmoidActivation();
        TensorArrayRef forwardBatch(TensorArrayRef xs) override;
        TensorRef forward(TensorRef x, ArrayIndex i) override;
        void backward(TensorRef dx, ArrayIndex i) override;
        TensorArrayRef parameters() override;
    };

} // IdealNN

#endif //IDEALNN_SIGMOIDACTIVATION_H
