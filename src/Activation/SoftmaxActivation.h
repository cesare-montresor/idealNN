//
// Created by cesare on 22/02/23.
//

#ifndef IDEALNN_SOFTMAXACTIVATION_H
#define IDEALNN_SOFTMAXACTIVATION_H

#include <Activation/Activation.h>

namespace IdealNN {
    struct SoftmaxActivation:  public Activation {
        TensorArrayRef xs;
        TensorArrayRef activations;

        SoftmaxActivation();
        TensorArrayRef forwardBatch(TensorArrayRef xs) override;
        TensorRef forward(TensorRef x, ArrayIndex i) override;
        void backward(TensorRef dx, ArrayIndex i) override;
        TensorArrayRef parameters() override;
    };

} // IdealNN

#endif //IDEALNN_SOFTMAXACTIVATION_H
