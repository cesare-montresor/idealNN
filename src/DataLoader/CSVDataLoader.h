//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include "../Tensor.h"

struct CSVDataLoader{
    Tensor *dataset;
    CSVDataLoader(int batch_size, std::string path);
    Tensor getData();
};


#endif //IDEALNN_DATALOADER_H
