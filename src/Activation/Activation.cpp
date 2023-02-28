//
// Created by cesare on 22/02/23.
//

#include <Activation/Activation.h>
#include <Utils.h>

namespace IdealNN {

    TensorArrayRef Activation::parameters() {
        return Utils::MakeTensorArray();
    }
}
