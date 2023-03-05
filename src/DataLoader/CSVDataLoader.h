//
// Created by cesare on 09/02/23.
//

#ifndef IDEALNN_CSVDATALOADER_H
#define IDEALNN_CSVDATALOADER_H

#include <DataLoader/DataLoader.h>
#include <Common.h>

namespace IdealNN {

    /// Class for loading CVS data
    struct CSVDataLoader: public DataLoader {
    protected:
        /// Holds the whole dataset
        TensorArray rows;

        /// Counter that points to the next batch of data
        ArrayIndex current = 0;

        /// Store the size of the mini-batch
        ArraySize batch_size;

        /// Number of columns found inside the CSV file.
        ArraySize col_nums;

        /// Random number generator used for shuffling the data.
        random_engine rndEngine{};

    public:

        /// Constructor for CSVDataloader
        /// @param batch_size number of element return for a single batch.
        /// @param path Absolute path to the CSV file.
        CSVDataLoader(int batch_size, const string &path);

        /// Reset the internal counter.
        void rewind();

        /// Shuffle the rows and reset the internal counter.
        void shuffle();

        /// Return the total number of rows.
        ArraySize numRows();

        /// Returns the next batch of data, if no more data is available, returns an empty vector.
        TensorArrayRef getData() override;

        static void splitXY(TensorArrayRef &batch, TensorArrayRef &xs, TensorArrayRef &ys, ArrayIndex xs_col_start, ArrayIndex xs_col_count,  ArrayIndex ys_col_start, ArrayIndex ys_col_count );

    };
}

#endif //IDEALNN_CSVDATALOADER_H
