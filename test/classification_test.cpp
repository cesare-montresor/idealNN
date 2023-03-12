
#include <Layer/LinearLayer.h>
#include <DataLoader/CSVDataLoader.h>
#include <Utils.h>
#include <Loss/CrossEntropyLoss.h>
#include <Optimizer/SDGOptimizer.h>
#include <Activation/SoftmaxActivation.h>
#include <Activation/TanhActivation.h>
#include <Activation/SigmoidActivation.h>

#include <catch2/catch.hpp>
#include <iostream>

namespace IdealNN {

    TEST_CASE("Classification: train with SDG on multiple epochs") {
        srand(0);
        std::cout << "Classification: train with SDG on multiple epochs" << std::endl << std::flush;
        auto learning_rate = 0.001;
        auto batch_size = 30;
        auto path = "/home/cesare/Projects/idealNN/extra/iris/IRIS.norm.csv";
        auto dl = CSVDataLoader::MakeCSVDataLoader(batch_size, path);

        auto xs = Tensor::MakeTensorArray();
        auto ys = Tensor::MakeTensorArray();

        auto fc1 = LinearLayer::MakeLinearLayer(4, 10);
        auto act1 = TanhActivation::MakeTanhActivation();
        auto fc2 = LinearLayer::MakeLinearLayer(10, 3);
        auto act2 = TanhActivation::MakeTanhActivation();
        auto softmax = SoftmaxActivation::MakeSoftmaxActivation();

        auto criterion = CrossEntropyLoss::MakeCrossEntropyLoss();

        auto trainLayers = Utils::MakeLayerArray();
        trainLayers->push_back(fc1);
        trainLayers->push_back(fc2);


        auto optimizer = SDGOptimizer::MakeSDGOptimizer(trainLayers, learning_rate);


        auto epoch = 0;
        auto epoch_max = 30; // with valgrind it takes too long
        //auto epoch_max = 3;
        auto num_batches = 0;

        ScalarValue initialLoss=0;
        ScalarValue finalLoss=0;
        ScalarValue loss;
        ScalarValue epochLoss=0;

        dl->shuffle();
        while(true) {
            auto batch = dl->getData();
            auto bs = Utils::getSize(batch);
            if(bs == 0){
                std::cout << "[EPOCH \t" << epoch << "]" << " --- " << "AVG Loss: ";
                std::cout << epochLoss / num_batches << " --- " << std::endl;
                if(epoch < epoch_max){
                    if (epoch == 0){
                        initialLoss = (epochLoss / num_batches);
                    }
                    epochLoss = 0;
                    num_batches = 0;
                    ++epoch;
                    dl->shuffle();
                    continue;
                }else{
                    finalLoss = (epochLoss / num_batches);
                    break;
                }
            }
            CSVDataLoader::splitXY(batch, xs, ys, 0, 4,  4, 3 );

            auto x1 = fc1->forwardBatch(xs);
            auto a1 = act1->forwardBatch(x1);
            auto x2 = fc2->forwardBatch(a1);
            auto a2 = act2->forwardBatch(x2);
            auto ys_hat = softmax->forwardBatch(a2);

            loss = criterion->loss(ys_hat,ys);
            epochLoss += loss;
            ++num_batches;

            criterion->backward();
            optimizer->step();
            optimizer->zero_grad();

        }

        REQUIRE( initialLoss > finalLoss );
    }


} // namespace IdealNN

