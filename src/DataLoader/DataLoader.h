//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include <Tensor/Tensor.h>
#include <Common.h>


namespace IdealNN {
    struct DataLoader {
        virtual TensorArrayRef getData() = 0;
    };

}
#endif //IDEALNN_DATALOADER_H
