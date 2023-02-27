//
// Created by cesare on 20/02/23.
//

#ifndef IDEALNN_OPTIMIZER_H
#define IDEALNN_OPTIMIZER_H

#include <Common.h>
#include <Loss/Loss.h>

namespace IdealNN{
    struct Optimizer {
        ScalarValue learning_rate;
        TensorArrayRef parameters;
        virtual void step()=0;
        virtual void zero_grad()=0;
    };
}

#endif //IDEALNN_OPTIMIZER_H
