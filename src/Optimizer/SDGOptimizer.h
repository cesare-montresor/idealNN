//
// Created by cesare on 20/02/23.
//

#ifndef IDEALNN_SDGOPTIMIZER_H
#define IDEALNN_SDGOPTIMIZER_H

#include <Optimizer/Optimizer.h>
#include <Common.h>

namespace IdealNN{
    /// Simple implementation of SDG, Stochastic Gradient Descent
    struct SDGOptimizer final: public Optimizer {
        /// Constructor for SGDOptimizer
        /// @param layers List of layers to be optimized ( their internal parameters )
        /// @param learning_rate Initial learning rate
        SDGOptimizer(const LayerArrayRef& layers, ScalarValue learning_rate);

        /// Constructor for SGDOptimizer
        /// @param params List of tensors to be optimized.
        /// @param learning_rate Initial learning rate
        SDGOptimizer(const TensorArrayRef& params, ScalarValue learning_rate);

        ///Apply the accumulated gradients corrections to the parameters.
        void step() override;

        ///Reset the accumulated gradients of all parameters.
        void zero_grad() override;
    };
}


#endif //IDEALNN_SDGOPTIMIZER_H
