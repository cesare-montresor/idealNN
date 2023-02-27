//
// Created by cesare on 08/02/23.
//

#include <Tensor/Tensor.h>
#include <Utils.h>
#include <Common.h>


//Tensor

namespace IdealNN {

    //Template args
    TensorRef Tensor::MakeTensor() { return std::make_shared<Tensor>(); }
    TensorRef Tensor::MakeTensor(ArraySize rows, ArraySize cols) { return std::make_shared<Tensor>(rows, cols); }
    TensorRef Tensor::MakeTensor(Tensor const &tensor) { return std::make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(TensorRef tensor) { return std::make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(Matrix const &matrix) { return std::make_shared<Tensor>(matrix); }
    TensorRef Tensor::MakeTensor(MatrixRef matrix) { return std::make_shared<Tensor>(matrix); }

    Tensor::Tensor():Tensor(0,0) {}

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
        this->data = matrix;
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


//Scalar
namespace IdealNN
{
    ScalarRef Scalar::MakeScalar(){return std::make_shared<Scalar>(); }
    ScalarRef Scalar::MakeScalar(ScalarValue value){ return std::make_shared<Scalar>(value);}
    ScalarRef Scalar::MakeScalar(Scalar const &scalar){ return std::make_shared<Scalar>(scalar); }
    ScalarRef Scalar::MakeScalar(ScalarRef scalar){ return std::make_shared<Scalar>(scalar); }
    ScalarRef Scalar::MakeScalar(Tensor const &tensor){ return std::make_shared<Scalar>(tensor); }
    ScalarRef Scalar::MakeScalar(TensorRef tensor){ return std::make_shared<Scalar>(tensor);}
    ScalarRef Scalar::MakeScalar(Matrix const &matrix){ return std::make_shared<Scalar>(matrix);}
    ScalarRef Scalar::MakeScalar(MatrixRef matrix){ return std::make_shared<Scalar>(matrix);}

    Scalar::Scalar(){
        this->data = Utils::MakeMatrix(1,1);
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }

    Scalar::Scalar(Scalar const &scalar){
        this->data = scalar.data;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }

    Scalar::Scalar(ScalarRef scalar){
        this->data = scalar->data;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }

    Scalar::Scalar(ScalarValue value){
        this->data = Utils::MakeMatrix(1,1);
        this->data->coeffRef(0) = value;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }

    Scalar::Scalar(Tensor const &tensor){
        assert(tensor.data->rows() == 1 && tensor.data->cols() == 1 );
        this->data = tensor.data;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }

    Scalar::Scalar(TensorRef tensor){
        assert(tensor->data->rows() == 1 && tensor->data->cols() == 1 );
        this->data = tensor->data;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }
    Scalar::Scalar(MatrixRef matrix){
        assert(matrix->rows() == 1 && matrix->cols() == 1 );
        this->data = matrix;
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }
    Scalar::Scalar(Matrix const &matrix){
        assert(matrix.rows() == 1 && matrix.cols() == 1 );
        this->data = Utils::MakeMatrix(matrix);
        this->operations = Utils::MakeLayerArray();
        this->gradients = Utils::MakeMatrix(1,1);
    }

    ScalarValue Scalar::value(){
        return (ScalarValue)this->data->coeff(0);
    }

    void Scalar::value(ScalarValue scalar){
        this->data->coeffRef(0,0) = scalar;
    }

    CoeffRef Scalar::val(){
        return this->data->coeffRef(0);
    }


}