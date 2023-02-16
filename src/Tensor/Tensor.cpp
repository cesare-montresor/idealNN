//
// Created by cesare on 08/02/23.
//

#include "Tensor.h"
#include "../Utils.h"

namespace IdealNN {
    TensorArrayRef Tensor::MakeTensorArray(){ return make_shared<TensorArray>(); }
    TensorArrayRef Tensor::MakeTensorArray(ArraySize size){ return make_shared<TensorArray>(size); }
    TensorArrayRef Tensor::MakeTensorArray(TensorArray tensorArray){ return make_shared<TensorArray>(tensorArray); }


    TensorRef Tensor::MakeTensor(){ return make_shared<Tensor>(); }
    TensorRef Tensor::MakeTensor(ArraySize rows, ArraySize cols){ return make_shared<Tensor>(rows, cols); }
    TensorRef Tensor::MakeTensor(Tensor const &tensor){ return make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(TensorRef tensor){ return make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(Matrix const &matrix){ return make_shared<Tensor>(matrix); }
    TensorRef Tensor::MakeTensor(MatrixRef matrix){ return make_shared<Tensor>(matrix); }


    Tensor::Tensor(){ this->data = Utils::MakeMatrix(0,0); }
    Tensor::Tensor(ArraySize rows, ArraySize cols ){ this->data = Utils::MakeMatrix(rows,cols); }
    Tensor::Tensor(Tensor const &tensor){ this->data = tensor.data; }
    Tensor::Tensor(TensorRef tensor){ this->data = tensor->data; }
    Tensor::Tensor(MatrixRef matrix){ this->data = matrix; }
    Tensor::Tensor(Matrix const &matrix){ this->data = make_shared<Matrix>(matrix); }


    TensorRef Tensor::view(ArraySize row_min, ArraySize col_min, ArraySize row_count, ArraySize col_count){
        auto view = data->block(row_min,col_min, row_count, col_count);
        return MakeTensor(view);
    }


}