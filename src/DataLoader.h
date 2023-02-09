//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include <memory>
#include "Tensor.h"

struct DataLoader;
using DataLoaderRef = std::unique_ptr<DataLoader>;

struct DataLoader {
    DataLoader(int batch_size);
    Tensor getData() = 0;
};


#endif //IDEALNN_DATALOADER_H
