//
// Created by cesare on 09/02/23.
//

#include "CSVDataLoader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "../Utils.h"

CSVDataLoader::CSVDataLoader(int batch_size, string fullpath ){
    this->batch_size = batch_size;
    this->rndEngine = default_random_engine{};

    ifstream file(fullpath);
    string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    stringstream ss(line);
    ScalarArray parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.push_back( Scalar( stof(&word[0])) );
    }
    uint cols = parsed_vec.size();

    file.seekg(0);

    // read the file
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            stringstream ss(line);
            data.push_back(new VectorRow(1, cols));
            uint i = 0;
            while (getline(ss, word, ',')) {
                data.back()->coeffRef(i) = Scalar( stof(&word[0]) );
                i++;
            }
        }
    }
    this->rewind();
}

void CSVDataLoader::rewind(){
    current=0;
}

ArraySize CSVDataLoader::numRows(){
    return data.size();
}

void CSVDataLoader::shuffle(){
    std::shuffle(data.begin(), data.end(), rndEngine);
    this->rewind();
}

VectorRowArray CSVDataLoader::getData(){
    auto batch = Utils::slice(data, current, batch_size);
    if(current + 1 >= data.size() ) {return VectorRowArray();}
    current += batch.size();
    return batch;
}