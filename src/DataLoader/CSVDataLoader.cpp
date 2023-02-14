//
// Created by cesare on 09/02/23.
//

#include "CSVDataLoader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "../Utils.h"

namespace IdealNN {

    CSVDataLoader::CSVDataLoader(int batch_size, string fullpath) {
        this->batch_size = batch_size;
        this->rndEngine = default_random_engine{};

        ifstream file(fullpath);
        string line, word;
        // determine number of columns in file
        getline(file, line, '\n');
        stringstream ss(line);
        ScalarArray parsed_vec;
        while (getline(ss, word, ',')) {
            parsed_vec.push_back(Scalar(stof(&word[0])));
        }
        col_nums = parsed_vec.size();

        file.seekg(0);

        // read the file
        if (file.is_open()) {
            while (getline(file, line, '\n')) {
                stringstream ss(line);
                auto row = Tensor::MakeTensor(1, col_nums);
                uint i = 0;
                while (getline(ss, word, ',')) {
                    row->data->row(0).coeffRef(i) = Scalar(stof(&word[0]));
                    i++;
                }
                rows.push_back( row );
            }

        }
        this->rewind();
    }

    void CSVDataLoader::rewind() {
        current = 0;
    }

    ArraySize CSVDataLoader::numRows() {
        return rows.size();
    }

    void CSVDataLoader::shuffle() {
        std::shuffle(rows.begin(), rows.end(), rndEngine);
        this->rewind();
    }

    TensorRef CSVDataLoader::getData() {
        auto batchRows = Utils::slice(rows,current,batch_size);
        auto numRows = batchRows.size();
        if(numRows == 0){return Tensor::MakeTensor(0,col_nums);}

        auto batch = Tensor::MakeTensor(numRows, col_nums );
        for(int i=0; i<numRows; i++){
            batch->data->row(i) = rows[current+i]->data->row(0);;
        }
        current+=numRows;

        return batch;
    }

}