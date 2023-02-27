//
// Created by Cesare on 25/10/2022.
//
#ifndef IDEALNN_SEQUENTIAL_H
#define IDEALNN_SEQUENTIAL_H

#include <Layer/Layer.h>

namespace IdealNN {
/**
 * Sequential thing
 */
    struct Sequential :  public  Layer {

        Sequential();

        void add(Layer *layer);
    };
}

#endif //IDEALNN_SEQUENTIAL_H
