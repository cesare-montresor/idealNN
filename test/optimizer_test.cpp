
#include <Layer/Dense.h>
#include <DataLoader/CSVDataLoader.h>
#include <Utils.h>
#include <Loss/MSELoss.h>
#include <Loss/CrossEntropyLoss.h>
#include <Optimizer/SDGOptimizer.h>
#include <Activation/SigmoidActivation.h>
#include <Activation/SoftmaxActivation.h>
#include <Activation/RELUActivation.h>

#include <catch2/catch.hpp>
#include <iostream>

namespace IdealNN {

    TEST_CASE("Optimizer: test SDG 10 epoch") {
        srand(0);

        auto learning_rate = 0.001;
        auto batch_size = 10;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.norm.csv";
        auto dl = new CSVDataLoader(batch_size, path);

        auto xs = Tensor::MakeTensorArray();
        auto ys = Tensor::MakeTensorArray();

        auto fc1 = Dense::MakeDense(4, 10);
        auto tanh1 = SigmoidActivation::MakeSigmoidActivation();
        auto fc2 = Dense::MakeDense(10, 3);
        auto softmax = SoftmaxActivation::MakeSoftmaxActivation();

        auto criterion = new CrossEntropyLoss();

        auto trainLayers = Utils::MakeLayerArray();
        trainLayers->push_back(fc1);
        trainLayers->push_back(fc2);


        auto optimizer = new SDGOptimizer(trainLayers, learning_rate);


        auto epoch = 0;
        auto epoch_max = 100;
        auto num_batches = 0;

        ScalarValue loss;
        ScalarValue epochLoss=0;

        dl->shuffle();
        while(true) {
            auto batch = dl->getData();
            auto bs = Utils::getSize(batch);
            if(bs == 0){
                std::cout << "---------------- " << "AVG Loss: " << epochLoss / num_batches << " ----------------" << std::endl;
                if(epoch < epoch_max){
                    epochLoss = 0;
                    num_batches = 0;
                    ++epoch;
                    dl->shuffle();
                    continue;
                }else{
                    break;
                }
            }
            CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 3 );

            auto x1 = fc1->forwardBatch(xs);
            auto a1 = tanh1->forwardBatch(x1);
            auto x2 = fc2->forwardBatch(a1);
            auto ys_hat = softmax->forwardBatch(x2);

            loss = criterion->loss(ys_hat,ys);
            epochLoss += loss;
            ++num_batches;

            std::cout << "Loss: " << loss << std::endl;
            //std::cout << fc2->weights->gradients->array().coeff(0) << " => " << std::flush;
            criterion->backward();
            optimizer->step();
            //std::cout << fc2->weights->gradients->array().coeff(0) << std::endl << std::flush;
            optimizer->zero_grad();

        }

        REQUIRE( loss < 2.82019f );
    }


    TEST_CASE("Optimizer: test SDG loss") {
        srand(0);

        auto learning_rate = 0.0001f;
        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto criterion = new MSELoss();

        auto fc1 = Dense::MakeDense(4, 10);
        auto fc2 = Dense::MakeDense(10, 1);
        auto layers = Utils::MakeLayerArray();
        layers->push_back(fc1);
        layers->push_back(fc2);


        auto optimizer = new SDGOptimizer(layers, learning_rate);

        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);
        auto xs = Tensor::MakeTensorArray(bs);
        auto ys = Tensor::MakeTensorArray(bs);
        CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 1 );


        auto x = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(x);
        auto loss = criterion->loss(ys_hat,ys);


        criterion->backward();
        optimizer->step();
        optimizer->zero_grad();

        auto x2 = fc1->forwardBatch(xs);
        auto ys_hat2 = fc2->forwardBatch(x);
        auto loss2 = criterion->loss(ys_hat2,ys);


        REQUIRE(Utils::ScalarValueEqual(loss, 2.82019f) );
    }
}
