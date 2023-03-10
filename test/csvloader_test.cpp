#include <Common.h>
#include <DataLoader/CSVDataLoader.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <Utils.h>

namespace IdealNN {
    
    TEST_CASE("CSVDataLoader: show batch") {
        srand(0);

        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/extra/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);
        std::cout << "Batch size: " << bs << std::endl;

        for (ArrayIndex i = 0; i < bs; i++) {
            //cout << "Item " << i << ": " << batch[i]->array() << endl;
        }
        delete dl;
        REQUIRE(true);
    }

    TEST_CASE("CSVDataLoader: show all batch") {
        srand(0);

        auto numRows = 150; //Hardcoded for iris dataset

        auto batch_size = 7;
        auto path = "/home/cesare/Projects/idealNN/extra/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        REQUIRE(numRows == dl->numRows());

        auto cnt = 0;

        while (true) {
            auto batch = dl->getData();
            auto bs = Utils::getSize(batch);
            if (bs == 0) {
                std::cout << "Final batch size: " << bs << std::endl;
                break;
            }
            for (ArrayIndex i = 0; i < bs; i++) {
                cnt++;
                //cout << "Item " << i << ": " << batch[i]->array() << endl;
            }
        }

        delete dl;
        REQUIRE(numRows == cnt);
    }

    TEST_CASE("CSVDataLoader: rewind") {
        auto numRows = 150; //Hardcoded for iris dataset

        auto batch_size = 7;
        auto path = "/home/cesare/Projects/idealNN/extra/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        REQUIRE(numRows == dl->numRows());

        auto cnt = 0;

        auto maxEpochs = 3;
        auto epochs = 0;

        std::cout << "Epoch 0: " << std::endl;
        while (true) {
            auto batch = dl->getData();
            auto bs = Utils::getSize(batch);
            if (bs == 0) {
                if (epochs < maxEpochs) {
                    std::cout << cnt << std::endl;
                    cnt = 0;
                    dl->rewind();
                    epochs++;
                    std::cout << "Epoch " << epochs << ": ";
                } else {
                    std::cout << cnt << std::endl;
                    std::cout << "Done!" << std::endl;
                    break;
                }

            }
            for (ArrayIndex i = 0; i < bs; i++) {
                cnt++;
                //cout << "Item " << i << ": " << batch[i]->array() << endl;
            }
        }

        delete dl;
        REQUIRE(numRows == cnt);
    }

    TEST_CASE("CSVDataLoader: shuffle") {
        auto class_idx = 4;
        auto batch_size = 50; //
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        dl->shuffle();
        auto batch = dl->getData();
        auto bs = Utils::getSize(batch);
        auto found = false;

        for (ArrayIndex i = 0; i < bs; i++) {
            auto row = batch->at(i)->data->array();
            auto cls = int(row(class_idx));
            if (cls >= 1) {
                std::cout << "Item " << i << ": " << row << std::endl;
                found = true;
                break;
            }

        }
        delete dl;
        REQUIRE(found);
    }

}

