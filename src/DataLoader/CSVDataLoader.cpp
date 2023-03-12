//
// Created by cesare on 09/02/23.
//

#include <DataLoader/CSVDataLoader.h>
#include <Common.h>
#include <Utils.h>
#include <Tensor/Tensor.h>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace IdealNN {

    CSVDataLoader::CSVDataLoader(int batch_size, const string &fullpath){
        rows = Tensor::MakeTensorArray();
        this->batch_size = batch_size;

        auto rowDelim = '\n';
        auto colDelim = ',';

        std::ifstream file(fullpath);
        if(!file.good()){
            std::cout << "The file as path " << fullpath << " does not exists" << std::endl << std::flush;
            assert(false);
        }
        string line, word;
        // determine number of columns in file
        getline(file, line, rowDelim);
        std::stringstream ss1(line);
        ScalarValueArray parsed_vec;
        while (getline(ss1, word, colDelim)) {
            parsed_vec.push_back(ScalarValue(std::stod(&word[0])));
        }
        col_nums = Utils::getSize(parsed_vec);

        file.seekg(0);

        // read the file
        if (file.is_open()) {
            while (getline(file, line, rowDelim)) {
                std::stringstream ss2(line);
                auto row = Tensor::MakeTensor(1,col_nums);
                ArrayIndex i = 0;
                while (getline(ss2, word, colDelim)) {
                    row->data->col(0).coeffRef(i) = ScalarValue( std::stod(&word[0]) );
                    i++;
                }
                rows->push_back( row );
            }

        }
        rewind();
    }

    void CSVDataLoader::rewind() {
        current = 0;
    }

    ArraySize CSVDataLoader::numRows() {
        return Utils::getSize(rows);
    }

    void CSVDataLoader::shuffle() {
        std::shuffle(rows->begin(), rows->end(), rndEngine);
        rewind();
    }

    TensorArrayRef CSVDataLoader::getData() {
        auto batchRows = Utils::slice(rows,current,batch_size);
        current += Utils::getSize(batchRows);
        return batchRows;
    }

    void CSVDataLoader::splitXY(const TensorArrayRef &batch,const  TensorArrayRef &xs,const  TensorArrayRef &ys, ArrayIndex xs_col_start, ArrayIndex xs_col_count,  ArrayIndex ys_col_start, ArrayIndex ys_col_count ){
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