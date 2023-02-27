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
        auto bs = batch->size();


        auto xs = Utils::MakeTensorArray(bs);
        auto ys = Utils::MakeTensorArray(bs);
        auto errors = Utils::MakeScalarArray(bs);

        for(int i =0 ; i< bs; i++){
            xs->at(i) = batch->at(i)->view(0,0,4,1);
            ys->at(i) = batch->at(i)->view(4,0,1,1);
        }

        auto fc1 = Utils::MakeDense(4, 10);
        auto fc2 = Utils::MakeDense(10, 1);

        auto a1s = fc1->forwardBatch(xs);
        auto ys_hat = fc2->forwardBatch(a1s);

        for(int i =0 ; i< bs; i++){
            auto y = (*ys)[i];
            auto y_hat = (*ys_hat)[i];
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            cout << "Error[" << i << "]: " << error << endl;
            errors->at(i) = Scalar::MakeScalar(error);
        }

        REQUIRE(Utils::ScalarValueEqual(errors->at(0)->value(), -1.76887f) );
    }


    TEST_CASE("Forward: 1 layer") {
        srand(0);
        auto batch_size = 3;
        auto path = "/home/cesare/Projects/idealNN/data/iris/IRIS.csv";
        auto dl = new CSVDataLoader(batch_size, path);
        auto batch = dl->getData();
        auto bs = batch->size();

        auto xs = Utils::MakeTensorArray(bs);
        auto ys = Utils::MakeTensorArray(bs);
        auto errors = Utils::MakeScalarArray(bs);

        for(int i =0 ; i< batch->size(); i++){
            xs->at(i) = batch->at(i)->view(0,0,4,1);
            ys->at(i) = batch->at(i)->view(4,0,1,1);
        }

        auto fc1 = Utils::MakeDense(4, 1);
        auto ys_hat = fc1->forwardBatch(xs);

        for(int i =0 ; i< batch->size(); i++){
            auto y = (*ys)[i];
            auto y_hat = (*ys_hat)[i];
            auto error = y->data->coeff(0) - y_hat->data->coeff(0);
            cout << "Error[" << i << "]: " << error << endl;
            errors->at(i) = Scalar::MakeScalar(error);
        }

        REQUIRE(Utils::ScalarValueEqual(errors->at(0)->value(), -4.46594f) );
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