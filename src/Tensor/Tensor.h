//
// Created by cesare on 08/02/23.
//

#ifndef IDEALNN_TENSOR_H
#define IDEALNN_TENSOR_H


#include <Eigen/Dense>
#include <Common.h>



namespace IdealNN {

    /*
    template<typename ... Args>
    inline TensorRef make_tensor(Args&& ... args) {
        return std::make_shared<Tensor>(args...);
    }
    */

    struct Tensor {

        //static
        /*
        static TensorRef MakeTensor();
        static TensorRef MakeTensor(ArraySize in, ArraySize out);
        static TensorRef MakeTensor(Tensor const &tensor);
        static TensorRef MakeTensor(TensorRef tensor);
        static TensorRef MakeTensor(MatrixRef matrix);
        static TensorRef MakeTensor(Matrix const &matrix);
        */

        template<typename ... Args>
        static TensorRef MakeTensor(Args&& ... args) {
            return std::make_shared<Tensor>(args...);
        }



        //constructors
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

#endif //IDEALNN_TENSOR_H
