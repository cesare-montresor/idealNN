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
        /// Holds the whole dataset to reduce overhead at train time
        TensorArrayRef rows;

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

        /// Given a batch of data, it splits each row of the vectors in features and labels ( ground-truth )
        /// filling up the related vectors.
        /// @param batch Represent a batch of data, each element is a row vector, and contains both features and ground-truth.
        /// @param xs Represent a batch of data, each element is subset of the original row vector, containing only the features to train on.
        /// @param ys Represent a batch of data, each element is subset of the original row vector, containing only the ground-truth.
        /// @param xs_col_start Starting index of the features in the rows.
        /// @param xs_col_count Number of columns in the features rows.
        /// @param ys_col_start Starting index of the ground-truth in the rows.
        /// @param ys_col_count Number of columns in the ground-truth rows.
        static void splitXY(const TensorArrayRef &batch, const  TensorArrayRef &xs, const TensorArrayRef &ys, ArrayIndex xs_col_start, ArrayIndex xs_col_count,  ArrayIndex ys_col_start, ArrayIndex ys_col_count );

    };
}

#endif //IDEALNN_CSVDATALOADER_H
