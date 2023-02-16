//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include <memory>
#include "../Tensor/Tensor.h"

namespace IdealNN {
    struct DataLoader {
        DataLoader(int batch_size);

        virtual Tensor getData() = 0;
    };

}
#endif //IDEALNN_DATALOADER_H