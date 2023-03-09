//
// Created by cesare on 08/02/23.
//

#include <Layer/Dense.h>
#include <Tensor/Tensor.h>
#include <Utils.h>

namespace IdealNN {
    DenseRef Dense::MakeDense(int in, int out) { return std::make_shared<Dense>(in, out); }

    Dense::Dense(ArraySize in, ArraySize out):Layer() {
        this->in = in;
        this->out = out;

        weights = Tensor::MakeTensor(in, out);
        bias = Tensor::MakeTensor(1, out);


        weights->initKaiming(in, 1, sqrt(3.0));
        bias->initKaiming(in, 1, 1);
        weights->zero_grad();
        bias->zero_grad();
    }

    TensorRef Dense::forward(TensorRef x) {
        auto result = ( (*x->data) * (*weights->data) + (*bias->data) );
        auto output = Tensor::MakeTensor(result);

        output->operation = shared_from_this();
        return output;
    }

    void Dense::backward(TensorRef dx, ArrayIndex i) {
        auto x = inputs->at(i); // inputs is the batch, x is the single input
        auto result = ( (*x->data) * (*weights->data) + (*bias->data) );

        //bias
        if(bias->use_grads) {
            (*bias->gradients) += (*dx->data);
        }

        //weights
        auto dense_dx = (*dx->data) * weights->data->transpose();
        auto weights_dx = result.transpose() * dense_dx;
        if(weights->use_grads) {
            (*weights->gradients) += weights_dx.transpose(); /// should I ?
        }

        if(x->operation) {
            auto next_dx = Tensor::MakeTensor( dense_dx );
            x->operation->backward(next_dx, i);
        }
    }

    TensorArrayRef Dense::parameters() {
        auto params = Tensor::MakeTensorArray();
        params->push_back(weights);
        params->push_back(bias);
        return params;
    }
}

/* https://iamtrask.github.io/2015/07/12/basic-python-network/

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

 */
