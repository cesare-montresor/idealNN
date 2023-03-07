//
// Created by cesare on 22/02/23.
//

#include <Activation/Activation.h>
#include <Tensor/Tensor.h>

namespace IdealNN {

    TensorArrayRef Activation::parameters() {
        return Tensor::MakeTensorArray();
    }
}
