#include <catch2/catch.hpp>
#include <Layer/Dense.h>
#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <Utils.h>
#include <Loss/MSELoss.h>

namespace IdealNN {
    TEST_CASE("Loss: MSELoss") {
        srand(0);

        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);

        auto xs = Tensor::MakeTensorArray(bs);
        auto ys = Tensor::MakeTensorArray(bs);
        CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 1 );

        auto fc1 = Dense::MakeDense(4, 1);
        auto ys_hat = fc1->forwardBatch(xs);

        auto criterion = new MSELoss();
        auto loss = criterion->loss(ys_hat,ys);

        REQUIRE(Utils::ScalarValueEqual(loss, 19.0876f) );
    }
}
