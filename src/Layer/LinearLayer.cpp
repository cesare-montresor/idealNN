//
// Created by cesare on 08/02/23.
//

#include <Layer/LinearLayer.h>
#include <Tensor/Tensor.h>


namespace IdealNN {
    LinearLayerRef LinearLayer::MakeLinearLayer(int in, int out) { return std::make_shared<LinearLayer>(in, out); }

    LinearLayer::LinearLayer(ArraySize in, ArraySize out){
        this->in = in;
        this->out = out;

        weights = Tensor::MakeTensor(in, out);
        bias = Tensor::MakeTensor(1, out);


        weights->initKaiming(in, 1, sqrt(3.0));
        bias->initKaiming(in, 1, 1);
        weights->zero_grad();
        bias->zero_grad();
    }

    TensorRef LinearLayer::forward(TensorRef x) {
        auto result = ( (*x->data) * (*weights->data) + (*bias->data) );
        auto output = Tensor::MakeTensor(result);

        output->operation = weak_from_this();
        return output;
    }

    void LinearLayer::backward(TensorRef dx, ArrayIndex i) {
        auto x = inputs->at(i); // inputs is the batch, x is the single input
        auto output = outputs->at(i);

        //bias
        if(bias->use_grads) {
            (*bias->gradients) += (*dx->data);
        }

        //weights
        auto dense_dx = (*dx->data) * weights->data->transpose();
        auto weights_dx = output->data->transpose() * dense_dx;
        if(weights->use_grads) {
            (*weights->gradients) += weights_dx.transpose(); /// should I ?
        }

        auto operation = x->operation.lock();
        if(operation){
            auto next_dx = Tensor::MakeTensor( dense_dx );
            operation->backward(next_dx,i);
        }
    }

    TensorArrayRef LinearLayer::parameters() {
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
