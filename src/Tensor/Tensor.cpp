//
// Created by cesare on 08/02/23.
//

#include <Tensor/Tensor.h>
#include <Utils.h>
#include <Common.h>

#include <utility>


//Tensor

namespace IdealNN {
    //static array
    TensorArrayRef Tensor::MakeTensorArray(){ return std::make_shared<TensorArray>(); }
    TensorArrayRef Tensor::MakeTensorArray(ArraySize size){ return std::make_shared<TensorArray>(size); }
    TensorArrayRef Tensor::MakeTensorArray(TensorArray tensorArray){ return std::make_shared<TensorArray>(std::move(tensorArray)); }



    TensorRef Tensor::MakeTensor(ArraySize rows, ArraySize cols) { return std::make_shared<Tensor>(rows, cols); }
    TensorRef Tensor::MakeTensor(Tensor &tensor) { return std::make_shared<Tensor>( std::move(tensor) ); }
    TensorRef Tensor::MakeTensor(const TensorRef& tensor) { return std::make_shared<Tensor>(tensor); }
    TensorRef Tensor::MakeTensor(const TensorData &tensorData) { return std::make_shared<Tensor>(tensorData); }
    TensorRef Tensor::MakeTensor(const TensorDataRef &tensorData) { return std::make_shared<Tensor>(tensorData); }



    Tensor::Tensor(ArraySize rows, ArraySize cols) {
        data = Utils::MakeTensorData(rows, cols);
        gradients = Utils::MakeTensorData(data->rows(), data->cols());
    }

    Tensor::Tensor(const Tensor &tensor){
        data = tensor.data;
        gradients = Utils::MakeTensorData(data->rows(), data->cols());
    }

    Tensor::Tensor(const TensorRef& tensor) {
        data = tensor->data;
        gradients = tensor->gradients;
        operation = tensor->operation;
        use_grads = tensor->use_grads;
    }

    Tensor::Tensor(TensorDataRef tensorData) {
        data = std::move(tensorData);
        gradients = Utils::MakeTensorData(data->rows(), data->cols());
    }

    Tensor::Tensor(const TensorData &tensorData) {
        data = Utils::MakeTensorData(tensorData);
        gradients = Utils::MakeTensorData(data->rows(), data->cols());
    }


    TensorRef Tensor::view(ArrayIndex row_min, ArrayIndex col_min, ArraySize row_count, ArraySize col_count) {
        auto view = data->block(row_min, col_min, row_count, col_count);
        return MakeTensor(view);
    }

    void Tensor::zero_grad() {
        gradients->setZero();
    }

    void Tensor::initZero(){ data->setZero(); }
    void Tensor::initUniform(){ data->setRandom(); }


    void Tensor::initKaiming(ArrayIndex fan_in, ScalarValue gain, ScalarValue scaleFactor){
        // https://pytorch.org/docs/stable/nn.init.html#torch-nn-init
        initUniform();
        auto fan = static_cast<ScalarValue>(fan_in);
        auto bound = (gain/sqrt(fan))*scaleFactor;
        (*data) = ( data->array() * bound ).matrix();
    }



}
