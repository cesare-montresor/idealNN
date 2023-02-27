//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H


#include <Eigen/Dense>
#include "../Common.h"



namespace IdealNN {

    /*
    template<typename ... Args>
    inline TensorRef make_tensor(Args&& ... args) {
        return std::make_shared<Tensor>(args...);
    }
    */

    struct Tensor {

        //static
        static TensorRef MakeTensor();
        static TensorRef MakeTensor(ArraySize in, ArraySize out);
        static TensorRef MakeTensor(Tensor const &tensor);
        static TensorRef MakeTensor(TensorRef tensor);
        static TensorRef MakeTensor(MatrixRef matrix);
        static TensorRef MakeTensor(Matrix const &matrix);

        /*
        template<typename ... Args>
        static TensorRef make_tensor(Args&& ... args) {
            return std::make_shared<Tensor>(args...);
        }
        */


        //constructors
        Tensor();
        Tensor(ArraySize rows, ArraySize cols);
        Tensor(Tensor const &tensor);
        explicit Tensor(TensorRef tensor);
        explicit Tensor(MatrixRef matrix);
        explicit Tensor(Matrix const &matrix);


        //Properties
        MatrixRef data;
        bool use_grads = true;
        LayerArrayRef operations;
        LayerRef operation;
        MatrixRef gradients;

        void zero_grad();

        //Methods
        TensorRef view(ArraySize row_min, ArraySize col_min, ArraySize row_count, ArraySize col_count);
        void inheritOperations(TensorRef tensor);
        void addOperation(LayerRef layer);
        void extendOperations(TensorRef tensor, LayerRef layer);
    };


}


namespace IdealNN {
    struct Scalar: public Tensor {
        //static
        static ScalarRef MakeScalar();
        static ScalarRef MakeScalar(ScalarValue value);
        static ScalarRef MakeScalar(Scalar const &scalar);
        static ScalarRef MakeScalar(ScalarRef scalar);
        static ScalarRef MakeScalar(Tensor const &tensor);
        static ScalarRef MakeScalar(TensorRef tensor);
        static ScalarRef MakeScalar(Matrix const &matrix);
        static ScalarRef MakeScalar(MatrixRef matrix);

        //constructors
        Scalar();
        explicit Scalar(ScalarValue value);
        Scalar(Scalar const &scalar);
        explicit Scalar(ScalarRef scalar);
        explicit Scalar(Tensor const &tensor);
        explicit Scalar(TensorRef tensor);
        explicit Scalar(Matrix const &matrix);
        explicit Scalar(MatrixRef matrix);

        ScalarValue value();
        void value(ScalarValue value);
        CoeffRef val();
    };
}

#endif //IDEALNN_TENSOR_H
