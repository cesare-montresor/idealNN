//
// Created by Cesare on 25/10/2022.
//
#ifndef IDEALNN_SEQUENTIAL_HPP
#define IDEALNN_SEQUENTIAL_HPP

#include "Layers/layer.hpp"

#include <list>

struct sequential {
    std::list<layer> layers;
    sequential();
    void add(layer layer);
};


#endif //IDEALNN_SEQUENTIAL_HPP
