//
// Created by cesare on 20/02/23.
//

#ifndef IDEALNN_SDGOPTIMIZER_H
#define IDEALNN_SDGOPTIMIZER_H

#include "Optimizer.h"
#include "../Common.h"

namespace IdealNN{
    struct SDGOptimizer: public Optimizer {
        SDGOptimizer(const LayerArrayRef& layers, ScalarValue learning_rate);
        SDGOptimizer(const TensorArrayRef& params, ScalarValue learning_rate);
        void step() override;
        void zero_grad() override;
    };
}


#endif //IDEALNN_SDGOPTIMIZER_H
