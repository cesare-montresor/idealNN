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

    protected:
        /// Provides an overridable default destructor, in case some derived class would need a more complex destruction logic.
        virtual ~DataLoader() = default;
    public:
        /// Returns the next batch of data, if no more data is available, returns an empty vector.
        virtual TensorArrayRef getData() = 0;
    };

}
#endif //IDEALNN_DATALOADER_H
