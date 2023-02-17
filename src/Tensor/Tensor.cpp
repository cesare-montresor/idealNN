//
// Created by cesare on 08/02/23.
//

#include "Tensor.h"
#include "../Utils.h"

namespace IdealNN {


    TensorRef Tensor::MakeTensor(){ return make_shared<Tensor>(); }
    TensorRef Tensor::MakeTensor(ArraySize rows, ArraySize cols){ return make_shared<Tensor>(rows, cols); }
    TensorRef Tensor::MakeTensor(Tensor const &tensor){ return make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(TensorRef tensor){ return make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(Matrix const &matrix){ return make_shared<Tensor>(matrix); }
    TensorRef Tensor::MakeTensor(MatrixRef matrix){ return make_shared<Tensor>(matrix); }


    Tensor::Tensor(){ this->data = Utils::MakeMatrix(0,0);}
    Tensor::Tensor(ArraySize rows, ArraySize cols ){
        this->data = Utils::MakeMatrix(rows,cols);
        this->operations = Utils::MakeLayerArray();
    }
    Tensor::Tensor(Tensor const &tensor){
        this->data = tensor.data;
        this->operations = Utils::MakeLayerArray();
    }
    Tensor::Tensor(TensorRef tensor){
        this->data = tensor->data;
        this->operations = Utils::MakeLayerArray();
    }
    Tensor::Tensor(MatrixRef matrix){
        this->data = matrix;
        this->operations = Utils::MakeLayerArray();
    }
    Tensor::Tensor(Matrix const &matrix){
        this->data = make_shared<Matrix>(matrix);
        this->operations = Utils::MakeLayerArray();
    }


    TensorRef Tensor::view(ArraySize row_min, ArraySize col_min, ArraySize row_count, ArraySize col_count){
        auto view = data->block(row_min,col_min, row_count, col_count);
        return MakeTensor(view);
    }

    void Tensor::inheritOperations(TensorRef tensor){
        operations = tensor->operations;
    }
    void Tensor::addOperation(LayerRef layer){
        operations->push_back(layer);
    }

    void Tensor::extendOperations(TensorRef tensor, LayerRef layer) {
        this->inheritOperations(tensor);
        this->addOperation(layer);
    }

}