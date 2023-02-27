#include "catch2/catch.hpp"
#include "Layer/Dense.h"
#include "DataLoader/CSVDataLoader.h"
#include <iostream>
#include "Utils.h"
#include "Loss/MSELoss.h"
#include "Loss/CrossEntropyLoss.h"
#include "Optimizer/SDGOptimizer.h"
#include "Activation/SigmoidActivation.h"
#include "Activation/RELUActivation.h"

namespace IdealNN {

    TEST_CASE("Optimizer: test SDG 10 epoch") {
        srand(0);

        auto learning_rate = 0.00000001f;
        auto batch_size = 10;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.norm.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        dl->shuffle();
        auto criterion = new CrossEntropyLoss();

        auto fc1 = Utils::MakeDense(4, 10);
        auto sig1 = Utils::MakeSigmoidActivation();
        auto fc2 = Utils::MakeDense(10, 3);
        auto softmax = Utils::MakeSigmoidActivation();

        auto layers = Utils::MakeLayerArray();
        layers->push_back(fc1);
        layers->push_back(fc2);


        auto optimizer = new SDGOptimizer(layers, learning_rate);

        auto epoch = 0;
        auto epoch_max = 10;
        ScalarValue loss;
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

            cout << "Loss: " << loss << endl;
            criterion->backward();
            optimizer->step();
            optimizer->zero_grad();
        }

        REQUIRE( loss < 2.82019f );
    }


    TEST_CASE("Optimizer: test SDG loss") {
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
        cout<< "Loss1: " << loss << endl;
        cout<< "Loss2: " << loss2 << endl;

        REQUIRE(Utils::ScalarValueEqual(loss, 2.82019f) );
    }
}

/*

 transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.CSVDataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs)
        break
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

*/