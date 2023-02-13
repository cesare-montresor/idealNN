//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include "../Tensor/Tensor.h"
#include <random>

struct CSVDataLoader{
    VectorRowArray data;
    ArrayIndex current=0;
    ArraySize batch_size;

    CSVDataLoader(int batch_size, string path);


    void rewind();
    void shuffle();
    ArraySize numRows();
    VectorRowArray getData();

private:
    default_random_engine rndEngine;
};


#endif //IDEALNN_DATALOADER_H
