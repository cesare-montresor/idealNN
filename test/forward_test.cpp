#include <catch2/catch.hpp>
#include <Layer/LinearLayer.h>
#include <DataLoader/CSVDataLoader.h>
#include <iostream>
#include <Utils.h>

namespace IdealNN {
    TEST_CASE("Forward: 2 layers") {
        srand(0);
        std::cout<<"Forward: 2 layers"<<std::endl<<std::flush;
        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/extra/iris/IRIS.csv";
        auto dl = CSVDataLoader::MakeCSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize( batch );


        auto xs = Tensor::MakeTensorArray(bs);
        auto ys = Tensor::MakeTensorArray(bs);
        auto errors = Utils::MakeScalarValueArray(bs);

        CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 1 );

        auto fc1 = LinearLayer::MakeLinearLayer(4, 10);
        auto fc2 = LinearLayer::MakeLinearLayer(10, 1);

        auto a1s = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(a1s);

        for(int i =0 ; i< bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            std::cout << "Error[" << i << "]: " << error << std::endl;
            errors->at(i) = error;
        }

        REQUIRE(Utils::ScalarValueEqual(errors->at(0), -0.565442) );
    }


    TEST_CASE("Forward: 1 layer") {
        srand(0);
        std::cout<<"Forward: 1 layers"<<std::endl<<std::flush;

        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/extra/iris/IRIS.csv";
        auto dl = CSVDataLoader::MakeCSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);

        auto xs = Tensor::MakeTensorArray(bs);
        auto ys = Tensor::MakeTensorArray(bs);
        auto errors = Utils::MakeScalarValueArray(bs);
        
        CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 1 );

        auto fc1 = LinearLayer::MakeLinearLayer(4, 1);
        auto ys_hat = fc1->forwardBatch(xs);

        for(ArrayIndex i =0 ; i< bs; i++){
            auto y = ys->at(i);
            auto y_hat = ys_hat->at(i);
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            std::cout << "Error[" << i << "]: " << error << std::endl;
            errors->at(i) = error;
        }

        REQUIRE(Utils::ScalarValueEqual(errors->at(0), -0.565442) );
    }
}
