//
// Created by Cesare on 25/10/2022.
//
#ifndef IDEALNN_SEQUENTIAL_H
#define IDEALNN_SEQUENTIAL_H

#include "Layer.h"

#include <list>


/**
 * Sequential thing
 */
struct Sequential: Layer {
    std::list<Layer> layers;
    bool compiled = false;

    Sequential();
    void add(Layer *layer);
    void compile();
};


#endif //IDEALNN_SEQUENTIAL_H
