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
        ScalarValueArray parsed_vec;
        while (getline(ss, word, ',')) {
            parsed_vec.push_back(ScalarValue(stof(&word[0])));
        }
        col_nums = parsed_vec.size();

        file.seekg(0);

        // read the file
        if (file.is_open()) {
            while (getline(file, line, '\n')) {
                stringstream ss(line);
                auto row = Tensor::MakeTensor(col_nums, 1);
                uint i = 0;
                while (getline(ss, word, ',')) {
                    row->data->col(0).coeffRef(i) = ScalarValue(stof(&word[0]));
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

    TensorArrayRef CSVDataLoader::getData() {
        auto batchRows = Utils::slice(rows,current,batch_size);
        current += batchRows.size();
        return Utils::MakeTensorArray(batchRows);
    }

}