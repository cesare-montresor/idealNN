//
// Created by cesare on 20/02/23.
//

#ifndef IDEALNN_OPTIMIZER_H
#define IDEALNN_OPTIMIZER_H

#include <Common.h>
#include <Loss/Loss.h>

namespace IdealNN{
    /// General purpose interface for any Optimizer object
    struct Optimizer {
    protected:
        virtual ~Optimizer() = default;

    public:

        ///Default learning rate
        ScalarValue learning_rate;
        ///Array of Tensors to be optimized
        TensorArrayRef parameters;
        ///Apply the accumulated gradients corrections to the parameters.
        virtual void step()=0;
        ///Reset the accumulated gradients.
        virtual void zero_grad()=0;
    };
}

#endif //IDEALNN_OPTIMIZER_H
