//
// Created by cesare on 08/02/23.
//

#include <Tensor/Tensor.h>
#include <Utils.h>
#include <Common.h>

#include <utility>


//Tensor

namespace IdealNN {

    TensorRef Tensor::MakeTensor(ArraySize rows, ArraySize cols) { return std::make_shared<Tensor>(rows, cols); }
    TensorRef Tensor::MakeTensor(Tensor &tensor) { return std::make_shared<Tensor>( std::move(tensor) ); }
    TensorRef Tensor::MakeTensor(const TensorRef& tensor) { return std::make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(const Matrix &matrix) { return std::make_shared<Tensor>(matrix); }
    TensorRef Tensor::MakeTensor(const MatrixRef& matrix) { return std::make_shared<Tensor>(matrix); }



    Tensor::Tensor(ArraySize rows, ArraySize cols) {
        this->data = Utils::MakeMatrix(rows, cols);
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(const Tensor &tensor){
        this->data = tensor.data;
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(const TensorRef& tensor) {
        this->data = tensor->data;
        this->gradients = tensor->gradients;
        this->operation = tensor->operation;
        this->use_grads = tensor->use_grads;
    }

    Tensor::Tensor(MatrixRef matrix) {
        this->data = std::move(matrix);
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }

    Tensor::Tensor(const Matrix &matrix) {
        this->data = Utils::MakeMatrix(matrix);
        this->gradients = Utils::MakeMatrix(data->rows(), data->cols());
    }


    TensorRef Tensor::view(ArrayIndex row_min, ArrayIndex col_min, ArraySize row_count, ArraySize col_count) {
        auto view = data->block(row_min, col_min, row_count, col_count);
        return MakeTensor(view);
    }



    void Tensor::zero_grad() {
        gradients->setZero();
    }
}
