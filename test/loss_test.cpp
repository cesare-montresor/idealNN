#include <catch2/catch.hpp>
#include <Layer/LinearLayer.h>
#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <Utils.h>
#include <Loss/MSELoss.h>

namespace IdealNN {
    TEST_CASE("Loss: MSELoss") {
        srand(0);
        std::cout<<"Loss: MSELoss"<<std::endl<<std::flush;

        auto batch_size = 3;
        auto path = "../../extra/iris/IRIS.csv";
        auto dl = CSVDataLoader::MakeCSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);

        auto xs = Tensor::MakeTensorArray(bs);
        auto ys = Tensor::MakeTensorArray(bs);
        CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 1 );

        auto fc1 = LinearLayer::MakeLinearLayer(4, 1);
        auto ys_hat = fc1->forwardBatch(xs);

        auto criterion = MSELoss::MakeMSELoss();
        auto loss = criterion->loss(ys_hat,ys);

        REQUIRE(Utils::ScalarValueEqual(loss, 19.0876f) );
    }
}
