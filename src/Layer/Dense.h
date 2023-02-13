//
// Created by cesare on 08/02/23.
//

#include "Layer.h"
#include "../Tensor/Tensor.h"

#ifndef IDEALNN_DENSE_H
#define IDEALNN_DENSE_H

//
// Created by Cesare on 25/10/2022.
//
struct Dense;
using DenseRef = unique_ptr<Dense>;

struct Dense: Layer{
    static DenseRef MakeDense(int in, int out);

    int in, out;
    MatrixRef weights;
    MatrixRef bias;
    MatrixRef gradients;
    MatrixRef activations;
    Dense(int in, int out);




    Tensor forward(const Tensor &input) override;
    MatrixRef forward(const MatrixRef input); // override;
    void backward() override;

};


#endif //IDEALNN_DENSE_H
