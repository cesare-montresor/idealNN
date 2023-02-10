//
// Created by cesare on 09/02/23.
//

#include "CSVDataLoader.h"
#include <iostream>
#include <fstream>

CSVDataLoader::CSVDataLoader(int batch_size, std::string path ){

    // the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
    // https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

    // the input is the file: "fileToOpen.csv":
    // a,b,c
    // d,e,f
    // This function converts input file data into the Eigen matrix format



    // the matrix entries are stored in this variable row-wise. For example if we have the matrix:
    // M=[a b c
    //    d e f]
    // the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
    // later on, this vector is mapped into the Eigen matrix format
    vector<double> matrixEntries;
    // in this object we store the data from the matrix
    ifstream matrixDataFile(path);
    // this variable is used to store the row of the matrix that contains commas
    string matrixRowString;
    // this variable is used to store the matrix entry;
    string matrixEntry;
    // this variable is used to track the number of rows
    int matrixRowNumber = 0;
    // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    while (getline(matrixDataFile, matrixRowString)){
        //convert matrixRowString that is a string to a stream variable.
        stringstream matrixRowStringStream(matrixRowString);
        // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        while (getline(matrixRowStringStream, matrixEntry, ',')){
            matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        matrixRowNumber++; //update the column numbers
    }

    // here we convet the vector variable into the matrix and return the resulting object,
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    auto mat = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);


}

Tensor CSVDataLoader::getData(){
    return Tensor( DimList{1,2,3} );
}