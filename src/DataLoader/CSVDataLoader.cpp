//
// Created by cesare on 09/02/23.
//

#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <Common.h>
#include <Utils.h>


namespace IdealNN {

    CSVDataLoader::CSVDataLoader(int batch_size, string fullpath){
        this->batch_size = batch_size;

        std::ifstream file(fullpath);
        string line, word;
        // determine number of columns in file
        getline(file, line, '\n');
        std::stringstream ss1(line);
        ScalarValueArray parsed_vec;
        while (getline(ss1, word, ',')) {
            parsed_vec.push_back(ScalarValue(std::stof(&word[0])));
        }
        col_nums = (ArraySize)parsed_vec.size();

        file.seekg(0);

        // read the file
        if (file.is_open()) {
            while (getline(file, line, '\n')) {
                std::stringstream ss2(line);
                auto row = Tensor::MakeTensor(col_nums, 1);
                uint i = 0;
                while (getline(ss2, word, ',')) {
                    row->data->col(0).coeffRef(i) = ScalarValue(std::stof(&word[0]));
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

    ArraySize CSVDataLoader::numRows() const {
        return (ArraySize)rows.size();
    }

    void CSVDataLoader::shuffle() {
        std::shuffle(rows.begin(), rows.end(), rndEngine);
        this->rewind();
    }

    TensorArrayRef CSVDataLoader::getData() {
        auto batchRows = Utils::slice(rows,current,batch_size);
        current += (ArraySize)batchRows.size();
        return  Utils::MakeTensorArray(std::move(batchRows));
    }

}