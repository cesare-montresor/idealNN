//
// Created by cesare on 20/02/23.
//

#ifndef IDEALNN_SDGOPTIMIZER_H
#define IDEALNN_SDGOPTIMIZER_H

#include "Optimizer.h"
#include "../Common.h"

namespace IdealNN{
    struct SDGOptimizer: Optimizer {

        SDGOptimizer(ScalarValue learning_rate);
        void step(Tensor tensor);
    };
}


#endif //IDEALNN_SDGOPTIMIZER_H
