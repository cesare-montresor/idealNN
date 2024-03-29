#include <catch2/catch.hpp>
#include <Layer/LinearLayer.h>
#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <Utils.h>
#include <Loss/MSELoss.h>

namespace IdealNN {
    TEST_CASE("Backward: test dense-linear") {
        std::srand(0);
        std::cout<<"Backward: test dense-linear"<<std::endl<<std::flush;

        auto batch_size = 3;
        auto path = "../../extra/iris/IRIS.csv";
        auto dl = CSVDataLoader::MakeCSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);

        auto xs = Tensor::MakeTensorArray(bs);
        auto ys = Tensor::MakeTensorArray(bs);
        CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 1 );

        auto fc1 = LinearLayer::MakeLinearLayer(4, 10);
        auto fc2 = LinearLayer::MakeLinearLayer(10, 1);

        auto x = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(x);

        auto criterion = MSELoss::MakeMSELoss();
        auto loss = criterion->loss(ys_hat,ys);

        std::cout << "Grads: FC1 " << fc1->weights->gradients->array() << std::endl;
        std::cout << "Grads: FC2 " << fc2->weights->gradients->array() << std::endl;

        criterion->backward();
        std::cout << "Loss: " << loss << std::endl;
        std::cout << "Grads: FC1 " << fc1->weights->gradients->array() << std::endl;
        std::cout << "Grads: FC2 " << fc2->weights->gradients->array() << std::endl;

        REQUIRE(Utils::ScalarValueEqual(loss, 2.82019f) );
    }
}
