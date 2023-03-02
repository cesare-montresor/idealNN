//
// Created by cesare on 08/02/23.
//

#include <Tensor/Tensor.h>
#include <Utils.h>
#include <Common.h>

#include <utility>


//Tensor

namespace IdealNN {

    //Template args
    /*
    TensorRef Tensor::MakeTensor() { return std::make_shared<Tensor>(); }
    TensorRef Tensor::MakeTensor(ArraySize rows, ArraySize cols) { return std::make_shared<Tensor>(rows, cols); }
    TensorRef Tensor::MakeTensor(Tensor const &tensor) { return std::make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(TensorRef tensor) { return std::make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(Matrix const &matrix) { return std::make_shared<Tensor>(matrix); }
    TensorRef Tensor::MakeTensor(MatrixRef matrix) { return std::make_shared<Tensor>(matrix); }
    */


    Tensor::Tensor(ArraySize rows, ArraySize cols) {
        this->data = Utils::MakeMatrix(rows, cols);
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(Tensor const &tensor){
        this->data = tensor.data;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(TensorRef tensor) {
        this->data = tensor->data;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(MatrixRef matrix) {
        this->data = std::move(matrix);
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(Matrix const &matrix) {
        this->data = Utils::MakeMatrix(matrix);
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }


    TensorRef Tensor::view(ArraySize row_min, ArraySize col_min, ArraySize row_count, ArraySize col_count) {
        auto view = data->block(row_min, col_min, row_count, col_count);
        return MakeTensor(view);
    }

    void Tensor::inheritOperations(TensorRef tensor) {
        operations = tensor->operations;
    }

    void Tensor::addOperation(LayerRef layer) {
        operations->push_back(layer);
    }



    void Tensor::extendOperations(TensorRef tensor, LayerRef layer) {
        this->inheritOperations(tensor);
        this->addOperation(layer);
    }

    void Tensor::zero_grad() {
        gradients->setZero();
    }
}
