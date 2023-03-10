//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_DATALOADER_H
#define IDEALNN_DATALOADER_H

#include <Tensor/Tensor.h>
#include <Common.h>


namespace IdealNN {
    /// Represent a general purpose interface for DataLoaders
    struct DataLoader {
        /// Returns the next batch of data, if no more data is available, returns an empty vector.
        virtual TensorArrayRef getData() = 0;
    protected:
        virtual ~DataLoader() = default;

    };

}
#endif //IDEALNN_DATALOADER_H
