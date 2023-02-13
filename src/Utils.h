//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include "Common.h"

struct Utils{

    static VectorRowArray slice(VectorRowArray array, ArrayIndex start, ArraySize count){
        if(start + 1 >= array.size() ) {return VectorRowArray();}
        auto items_count = start + count < array.size() ? count : (array.size() - start);

        auto first = array.begin() + start;
        auto last = array.begin() + start + items_count;

        return VectorRowArray(first, last);
    }


    static VectorRowRef MakeVectorRow(ArraySize size) { return make_unique<VectorRow>(size); }
    static VectorColRef MakeVectorCol(ArraySize size) { return make_unique<VectorCol>(size); }
    static MatrixRef MakeMatrix(ArraySize in, ArraySize out) { return make_unique<Matrix>(in, out); }
};

#endif //IDEALNN_UTILS_H

