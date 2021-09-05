# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import time

import deepdish as dd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("-N", default=10, type=int, help="number of datapoints")
args = parser.parse_args()


class GraspingDataset(Dataset):
    def __init__(self, fileNames, fields=[], transform=None, transformDepth=None):
        self.transform = transform
        self.transformDepth = transformDepth
        self.fileNames = fileNames
        self.fields = fields + ["label"]
        self.numGroup = 100  # data points per file

        self.dataList = None
        self.dataFileID = -1

    def __len__(self):
        return len(self.fileNames * self.numGroup)

    def load_data(self, idx):
        dirName = self.fileNames[idx]
        data = {}

        for k in self.fields:
            fn = dirName.split("/")[-1]

            fnk = "{}_{}.h5".format(fn, k)

            filenamek = os.path.join(dirName, fnk)
            d = dd.io.load(filenamek)

            data[k] = d

        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fileID = idx // self.numGroup
        if fileID != self.dataFileID:
            self.dataList = self.load_data(fileID)
            self.dataFileID = fileID

        sample = {}

        data = self.dataList

        for k in self.fields:
            d = data[k][idx % self.numGroup]

            if k in ["tactileColorL", "tactileColorR", "visionColor"]:
                d = d[:, :, :3]
                # print(k, d.min(), d.max())

            if k in ["tactileDepthL", "tactileDepthR", "visionDepth"]:
                d = np.dstack([d, d, d])

            if k in ["tactileDepthL", "tactileDepthR"]:
                d = d / 0.002 * 255
                d = np.clip(d, 0, 255).astype(np.uint8)
                # print("depth min", d.min(), "max", d.max())

            if k in ["visionDepth"]:
                d = (d * 255).astype(np.uint8)

            if k in [
                "tactileColorL",
                "tactileColorR",
                "visionColor",
                "visionDepth",
            ]:
                if self.transform:
                    d = self.transform(d)

            if k in [
                "tactileDepthL",
                "tactileDepthR",
            ]:
                # print("before", d.min(), d.max(), d.mean(), d.std())
                d = self.transformDepth(d)
                # d = (d + 2) / 0.05
                # print("after", d.min(), d.max(), d.mean(), d.std())

            sample[k] = d

        return sample


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Model(nn.Module):
    def __init__(self, fields):
        super(Model, self).__init__()

        self.fields = fields

        for k in self.fields:
            # Load base network
            net = self.get_base_net()
            net_name = "net_{}".format(k)

            # Add for training
            self.add_module(net_name, net)

        self.nb_feature = 512
        self.fc1 = nn.Linear(self.nb_feature * len(fields), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def get_base_net(self):
        # Load pre-trained resnet-18
        net = torchvision.models.resnet18(pretrained=True)

        # Remove the last fc layer, and rebuild
        modules = list(net.children())[:-1]
        net = nn.Sequential(*modules)

        return net

    def forward(self, x):
        features = []

        # Get stored modules/networks
        nets = self.__dict__["_modules"]

        for k in self.fields:
            # Get the network by name
            net_name = "net_{}".format(k)
            net = nets[net_name]

            # Forward
            embedding = net(x[k])
            embedding = embedding.view(-1, self.nb_feature)

            features.append(embedding)

        # Concatenate embeddings
        emb_fusion = torch.cat(features, dim=1)

        # Add fc layer for final prediction
        output = self.fc1(emb_fusion)
        output = self.fc2(F.relu(output))
        output = self.fc3(F.relu(output))

        return output

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class Learning:
    def __init__(self, K, i, fields=None):
        self.fields = fields

        self.build_model()
        self.load_data(K, i)

    def build_model(self):
        # Loading pre-trained model
        model = Model(self.fields)

        self.device = torch.device("cuda:0")
        model.to(self.device)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005)

    def load_data(self, K, i):
        # K-fold, test the i-th fold, train the rest

        # rootDir = "data/test/"
        # rootDir = "data/resmid/"
        rootDir = "/media/shawn/Extreme SSD/Code/stability/data/separate"
        # fileNames = glob.glob(os.path.join(rootDir, "*.h5"))
        fileNames = glob.glob(os.path.join(rootDir, "*"))
        fileNames = sorted(fileNames)[: args.N]
        # print(fileNames)

        # Split K fold
        N = len(fileNames)
        n = N // K

        idx = list(range(N))
        testIdx = idx[n * i : n * (i + 1)]
        trainIdx = list(set(idx) - set(testIdx))

        trainFileNames = [fileNames[i] for i in trainIdx]
        testFileNames = [fileNames[i] for i in testIdx]

        trainTransform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                AddGaussianNoise(0.0, 0.01),
            ]
        )

        trainTransformDepth = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1,), std=(0.2,)),
                AddGaussianNoise(0.0, 0.01),
            ]
        )

        # Create training dataset and dataloader
        trainDataset = GraspingDataset(
            trainFileNames,
            fields=self.fields,
            transform=trainTransform,
            transformDepth=trainTransformDepth,
        )
        trainLoader = torch.utils.data.DataLoader(
            trainDataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True
        )

        testTransform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

        testTransformDepth = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1,), std=(0.2,)),
                # AddGaussianNoise(0.0, 0.01),
            ]
        )

        # Create training dataset and dataloader
        testDataset = GraspingDataset(
            testFileNames,
            fields=self.fields,
            transform=testTransform,
            transformDepth=testTransformDepth,
        )
        testLoader = torch.utils.data.DataLoader(
            testDataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True
        )

        # tot = 0
        # suc = 0
        # for i, data in enumerate(trainLoader):

        #     x = {}
        #     for k in self.fields:
        #         x[k] = data[k].to(self.device).squeeze(0)
        #         print(k, x[k].size())

        #     label = data["label"].squeeze(0)
        #     print(label.size())
        #     suc += label.sum().item()
        #     tot += label.size(0)
        # print("ratio", suc / tot)

        self.trainLoader, self.testLoader = trainLoader, testLoader

    def evaluation(self):
        total, correct = 0, 0
        print("")

        self.model.eval()

        for i, data in enumerate(self.testLoader):

            x = {}
            for k in self.fields:
                x[k] = data[k].to(self.device)

            label = data["label"].to(self.device)

            with torch.no_grad():
                outputs = self.model(x)

                pred = outputs.argmax(axis=-1)

                total += label.size(0)
                correct += (pred == label).sum().item()
            print("\r Evaluation: ", i, correct / total, end=" ")

        acc = correct / total
        # print("val accuracy", acc)

        return acc

    def train(self, nbEpoch=10):

        lossTrainList = []
        accTestList = []

        nbatch = len(self.trainLoader)
        # self.evaluation()
        # accTestList.append(0.6)
        for epoch in range(nbEpoch):  # loop over the dataset multiple times

            running_loss = 0.0
            print("")
            st = time.time()

            self.model.train()

            for i, data in enumerate(self.trainLoader):

                x = {}
                for k in self.fields:
                    x[k] = data[k].to(self.device)

                label = data["label"].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.model(x)

                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                percent = 20 * i // nbatch
                progressBar = "#" * percent + "." * (20 - percent)
                passedTime = time.time() - st
                eta = passedTime / (i + 1) * (nbatch - i)
                print(
                    "\r%s [%d, %5d] loss: %.6f elapse: %.2fs eta: %.2fs"
                    % (
                        progressBar,
                        epoch + 1,
                        i + 1,
                        running_loss / 2000,
                        passedTime,
                        eta,
                    ),
                    end=" ",
                )

                # print statistics
                lossTrainList.append(loss.item())
                running_loss = (running_loss * i + loss.item()) / (i + 1)
                # if i % 10 == 9:  # print every 2000 mini-batches
                #     print(
                #         "[%d, %5d] loss: %.6f" % (epoch + 1, i + 1, running_loss / 2000)
                #     )
                #     running_loss = 0.0
            accTest = self.evaluation()
            accTestList.append(accTest)

            print(accTest, end=" ")
        return lossTrainList, accTestList


K = 5


def test(fields):
    N = args.N

    print("N =", N)
    logDir = "logs/grasp"
    os.makedirs(logDir, exist_ok=True)

    print(fields)
    accs = []
    # for i in range(K):
    for i in range(K):
        st = time.time()

        print("Fold {} of {}".format(i, K), end=" ")

        learner = Learning(K, i, fields=fields)
        lossTrainList, accTestList = learner.train(10)

        fn = "{}/field{}_N{}_i{}.h5".format(logDir, fields, N, i)

        dd.io.save(fn, {"lossTrainList": lossTrainList, "accTestList": accTestList})

        acc = learner.evaluation()
        print(acc, end=" ")

        print("time {:.3f}s".format(time.time() - st))
        accs.append(acc)

        modelDir = "models/grasp"
        os.makedirs(modelDir, exist_ok=True)
        fnModel = "{}/field{}_N{}_i{}.pth".format(modelDir, fields, N, i)

        learner.model.save(fnModel)

        # learner.model.load(fnModel)
        # acc = learner.evaluation()
        del learner

    print(accs)
    print("{:.2f}% ± {:.2f}%".format(np.mean(accs) * 100, np.std(accs) * 100))


fieldsList = [
    # ["tactileColorL", "tactileDepthL", "visionColor"],
    ["tactileColorL", "tactileColorR", "visionColor"],
    ["visionColor"],
    ["tactileColorL", "tactileColorR"],
    # ["tactileDepthL"],
    # ["tactileColorL", "visionDepth"],
    # ["tactileColorL", "tactileDepthL"],
]

for fields in fieldsList:
    test(fields)
