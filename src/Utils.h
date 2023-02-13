//
// Created by cesare on 12/02/23.
//

#ifndef IDEALNN_UTILS_H
#define IDEALNN_UTILS_H

#include "Common.h"

namespace IdealNN{
    namespace Utils{

        /*
        VectorRowArray slice(VectorRowArray array, ArrayIndex start, ArraySize count){
            if(start + 1 >= array.size() ) {return VectorRowArray();}
            auto items_count = start + count < array.size() ? count : (array.size() - start);

            auto first = array.begin() + start;
            auto last = array.begin() + start + items_count;

            return VectorRowArray(first, last);
        }
         */

        MatrixArray slice(MatrixArray array, ArrayIndex start, ArraySize count){
            if(start + 1 >= array.size() ) {return MatrixArray();}
            auto items_count = start + count < array.size() ? count : (array.size() - start);

            auto first = array.begin() + start;
            auto last = array.begin() + start + items_count;

            return MatrixArray(first, last);
        }

        /*
        VectorRowRef MakeVectorRow(ArraySize size) { return make_shared<VectorRow>(size); }
        VectorColRef MakeVectorCol(ArraySize size) { return make_shared<VectorCol>(size); }
        */
        MatrixRef MakeMatrix(ArraySize in, ArraySize out) { return make_shared<Matrix>(in, out); }
    }
}

#endif //IDEALNN_UTILS_H

