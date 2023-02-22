//
// Created by cesare on 20/02/23.
//

#ifndef IDEALNN_OPTIMIZER_H
#define IDEALNN_OPTIMIZER_H

#include "../Common.h"
#include "../Loss/Loss.h"

namespace IdealNN{
    struct Optimizer {
        ScalarValue learning_rate;
        virtual void step(Tensor tensor)=0;
    };
}

#endif //IDEALNN_OPTIMIZER_H
