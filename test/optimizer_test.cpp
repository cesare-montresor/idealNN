
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

        auto learning_rate = 0.000001f;
        auto batch_size = 20;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.norm.csv";
        auto dl = new CSVDataLoader(batch_size, path);



        auto fc1 = Utils::MakeDense(4, 10);
        auto sig1 = Utils::MakeSigmoidActivation();
        auto fc2 = Utils::MakeDense(10, 3);
        auto softmax = Utils::MakeSoftmaxActivation();

        auto criterion = new CrossEntropyLoss();

        auto trainLayers = Utils::MakeLayerArray();
        trainLayers->push_back(fc1);
        trainLayers->push_back(fc2);


        auto optimizer = new SDGOptimizer(trainLayers, learning_rate);

        auto epoch = 0;
        auto epoch_max = 10;
        ScalarValue loss;
        dl->shuffle();
        while(true) {
            auto batch = dl->getData();
            auto bs = batch->size();
            if(bs == 0){
                if(epoch < epoch_max){
                    epoch++;
                    dl->shuffle();
                    continue;
                }else{
                    break;
                }
            }

            auto xs = Utils::MakeTensorArray(bs);
            auto ys = Utils::MakeTensorArray(bs);

            for (int i = 0; i < bs; i++) {
                xs->at(i) = batch->at(i)->view(0, 0, 4, 1);
                ys->at(i) = batch->at(i)->view(4, 0, 3, 1);
            }

            auto x1 = fc1->forwardBatch(xs);
            auto a1 = sig1->forwardBatch(x1);
            auto x2 = fc2->forwardBatch(a1);
            auto ys_hat = softmax->forwardBatch(x2);

            loss = criterion->loss(ys, ys_hat);

            std::cout << "Loss: " << loss << std::endl;
            criterion->backward();
            optimizer->step();
            optimizer->zero_grad();
        }

        REQUIRE( loss < 2.82019f );
    }


    TEST_CASE("Optimizer: test SDG loss") {
        srand(0);

        auto learning_rate = 0.000001f;
        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto criterion = new MSELoss();

        auto fc1 = Utils::MakeDense(4, 10);
        auto fc2 = Utils::MakeDense(10, 1);
        auto layers = Utils::MakeLayerArray();
        layers->push_back(fc1);
        layers->push_back(fc2);


        auto optimizer = new SDGOptimizer(layers, learning_rate);

        auto batch = dl->getData();
        auto bs = batch->size();

        auto xs = Utils::MakeTensorArray(bs);
        auto ys = Utils::MakeTensorArray(bs);

        for(int i =0 ; i<bs ; i++){
            xs->at(i) = batch->at(i)->view(0,0,4,1);
            ys->at(i) = batch->at(i)->view(4,0,1,1);
        }

        auto x = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(x);
        auto loss = criterion->loss(ys,ys_hat);


        criterion->backward();
        optimizer->step();
        optimizer->zero_grad();

        auto x2 = fc1->forwardBatch(xs);
        auto ys_hat2 = fc2->forwardBatch(x);
        auto loss2 = criterion->loss(ys,ys_hat2);
        std::cout<< "Loss1: " << loss << std::endl;
        std::cout<< "Loss2: " << loss2 << std::endl;

        REQUIRE(Utils::ScalarValueEqual(loss, 2.82019f) );
    }
}
