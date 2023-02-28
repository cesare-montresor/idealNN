#include <catch2/catch.hpp>
#include <Layer/Dense.h>
#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <Utils.h>
#include <Loss/MSELoss.h>

namespace IdealNN {
    TEST_CASE("Backward: test dense-linear") {
        srand(0);

        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = batch->size();

        auto xs = Utils::MakeTensorArray(bs);
        auto ys = Utils::MakeTensorArray(bs);

        for(int i =0 ; i<bs ; i++){
            xs->at(i) = batch->at(i)->view(0,0,4,1);
            ys->at(i) = batch->at(i)->view(4,0,1,1);
        }

        auto fc1 = Utils::MakeDense(4, 10);
        auto fc2 = Utils::MakeDense(10, 1);

        auto x = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(x);

        auto criterion = new MSELoss();
        auto loss = criterion->loss(ys,ys_hat);

        std::cout << "Grads: FC1 " << fc1->weights->gradients->array() << std::endl;
        std::cout << "Grads: FC2 " << fc2->weights->gradients->array() << std::endl;

        criterion->backward();
        std::cout << "Loss: " << loss << std::endl;
        std::cout << "Grads: FC1 " << fc1->weights->gradients->array() << std::endl;
        std::cout << "Grads: FC2 " << fc2->weights->gradients->array() << std::endl;

        REQUIRE(Utils::ScalarValueEqual(loss, 2.82019f) );
    }
}
