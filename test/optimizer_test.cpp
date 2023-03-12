
#include <Layer/LinearLayer.h>
#include <DataLoader/CSVDataLoader.h>
#include <Utils.h>
#include <Loss/MSELoss.h>
#include <Loss/CrossEntropyLoss.h>
#include <Optimizer/SDGOptimizer.h>
#include <Activation/SigmoidActivation.h>
#include <Activation/SoftmaxActivation.h>
#include <Activation/RELUActivation.h>
#include <Activation/TanhActivation.h>

#include <catch2/catch.hpp>
#include <iostream>

namespace IdealNN {

    TEST_CASE("Optimizer: test SDG loss") {
        srand(0);

        auto learning_rate = 0.0001f;
        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto criterion = new MSELoss();

        auto fc1 = LinearLayer::MakeLinearLayer(4, 10);
        auto fc2 = LinearLayer::MakeLinearLayer(10, 1);
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

        delete dl;
        delete criterion;
        delete optimizer;
        REQUIRE(Utils::ScalarValueEqual(loss, 2.82019f) );
    }
}
