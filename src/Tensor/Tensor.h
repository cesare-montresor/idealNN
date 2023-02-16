//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H


#include <Eigen/Dense>
#include "../Common.h"


namespace IdealNN {
    struct Tensor;

    using TensorRef = shared_ptr<Tensor>;
    typedef vector<TensorRef> TensorArray;
    typedef shared_ptr<TensorArray> TensorArrayRef;

    struct Tensor {

        //static array
        static TensorArrayRef MakeTensorArray();
        static TensorArrayRef MakeTensorArray(ArraySize size);
        static TensorArrayRef MakeTensorArray(TensorArray tensorArray);

        //static
        static TensorRef MakeTensor();
        static TensorRef MakeTensor(ArraySize in, ArraySize out);
        static TensorRef MakeTensor(Tensor const &tensor);
        static TensorRef MakeTensor(TensorRef tensor);
        static TensorRef MakeTensor(MatrixRef matrix);
        static TensorRef MakeTensor(Matrix const &matrix);


        //constructors
        Tensor();
        Tensor(ArraySize rows, ArraySize cols );
        Tensor(Tensor const &tensor);
        Tensor(TensorRef tensor);
        Tensor(MatrixRef matrix);
        Tensor(Matrix const &matrix);


        //Properties
        MatrixRef data;

        //Methods
        TensorRef view(ArraySize row_min, ArraySize col_min, ArraySize row_count, ArraySize col_count);


    };
}

#endif //IDEALNN_TENSOR_H
