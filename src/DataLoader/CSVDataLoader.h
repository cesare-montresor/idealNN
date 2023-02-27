//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_CSVDATALOADER_H
#define IDEALNN_CSVDATALOADER_H

#include <DataLoader/DataLoader.h>


namespace IdealNN {
    struct CSVDataLoader: public DataLoader {
        TensorArray rows;

        ArrayIndex current = 0;
        ArraySize batch_size;
        ArraySize col_nums;

        CSVDataLoader(int batch_size, string path);


        void rewind();

        void shuffle();

        ArraySize numRows() const;

        TensorArrayRef getData() override;

    private:
        std::default_random_engine rndEngine{};
    };
}

#endif //IDEALNN_CSVDATALOADER_H
