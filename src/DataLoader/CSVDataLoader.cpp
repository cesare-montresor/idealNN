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

    CSVDataLoader::CSVDataLoader(int batch_size, const string &fullpath){
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
                auto row = Tensor::MakeTensor(1,col_nums);
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

    ArraySize CSVDataLoader::numRows() {
        return (ArraySize) rows.size();
    }

    void CSVDataLoader::shuffle() {
        std::shuffle(rows.begin(), rows.end(), rndEngine);
        this->rewind();
    }

    TensorArrayRef CSVDataLoader::getData() {
        auto batchRows = Utils::slice(rows,current,batch_size);
        current += (ArraySize)batchRows.size();
        return  Tensor::MakeTensorArray(std::move(batchRows));
    }

    void CSVDataLoader::splitXY(TensorArrayRef &batch, TensorArrayRef &xs, TensorArrayRef &ys, ArrayIndex xs_col_start, ArrayIndex xs_col_count,  ArrayIndex ys_col_start, ArrayIndex ys_col_count ){
        auto b_size = Utils::getSize(batch);
        auto xs_size = Utils::getSize(xs);
        auto ys_size = Utils::getSize(ys);
        if( xs_size != b_size ){ xs->resize(b_size); }
        if( ys_size != b_size ){ ys->resize(b_size); }

        for(int i=0 ; i < b_size ; i++){
            xs->at(i) = batch->at(i)->view(0, xs_col_start, 1, xs_col_count);
            ys->at(i) = batch->at(i)->view(0, ys_col_start, 1, ys_col_count);
        }
    }


}