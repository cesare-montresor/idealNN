//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include "../Tensor/Tensor.h"
#include <random>

namespace IdealNN {
    struct CSVDataLoader {
        TensorArray rows;

        ArrayIndex current = 0;
        ArraySize batch_size;
        ArraySize col_nums;

        CSVDataLoader(int batch_size, string path);


        void rewind();

        void shuffle();

        ArraySize numRows();

        TensorArray getData();

    private:
        default_random_engine rndEngine;
    };
}

#endif //IDEALNN_DATALOADER_H
