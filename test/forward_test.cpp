#include <catch2/catch.hpp>
#include <Layer/Dense.h>
#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <Utils.h>

namespace IdealNN {
    TEST_CASE("Forward: 2 layers") {
        srand(0);

        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize( batch );


        auto xs = Utils::MakeTensorArray(bs);
        auto ys = Utils::MakeTensorArray(bs);
        auto errors = Utils::MakeScalarValueArray(bs);

        for(int i =0 ; i< bs; i++){
            xs->at(i) = batch->at(i)->view(0,0,4,1);
            ys->at(i) = batch->at(i)->view(4,0,1,1);
        }

        auto fc1 = Utils::MakeDense(4, 10);
        auto fc2 = Utils::MakeDense(10, 1);

        auto a1s = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(a1s);

        for(int i =0 ; i< bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            std::cout << "Error[" << i << "]: " << error << std::endl;
            errors->at(i) = error;
        }

        REQUIRE(Utils::ScalarValueEqual(errors->at(0), -1.76887f) );
    }


    TEST_CASE("Forward: 1 layer") {
        srand(0);
        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);

        auto xs = Utils::MakeTensorArray(bs);
        auto ys = Utils::MakeTensorArray(bs);
        auto errors = Utils::MakeScalarValueArray(bs);

        for(ArrayIndex i =0 ; i< bs; i++){
            xs->at(i) = batch->at(i)->view(0,0,4,1);
            ys->at(i) = batch->at(i)->view(4,0,1,1);
        }

        auto fc1 = Utils::MakeDense(4, 1);
        auto ys_hat = fc1->forwardBatch(xs);

        for(ArrayIndex i =0 ; i< bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            std::cout << "Error[" << i << "]: " << error << std::endl;
            errors->at(i) = error;
        }

        REQUIRE(Utils::ScalarValueEqual(errors->at(0), -4.46594f) );
    }
}
