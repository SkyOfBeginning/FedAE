import random
from typing import TypeVar, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import MNIST,CIFAR100
from torchvision.transforms import transforms
from tqdm import tqdm

import DataframeToDataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class cifar100_Data_Spliter():

    def __init__(self,client_num,task_num,feature_extractor):
        self.client_num = client_num
        self.task_num = task_num
        self.transform = transforms.Compose([transforms.RandomCrop((32,32), padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.transform1 = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.feature_extractor = feature_extractor


    def random_split(self):
        self.cifar100_dataset = CIFAR100(root='./dataset', train=True, download=True, transform=self.transform)
        # 特征提取
        trainloader = DataLoader(self.cifar100_dataset, batch_size=64, shuffle=True)
        new_test_data = []
        new_test_label = []
        with torch.no_grad():
            for x, y in tqdm(trainloader):
                self.feature_extractor.to("cuda")
                x = x.to("cuda")
                x, _ = self.feature_extractor(x)
                x = x.cpu()
                for i in range(len(x)):
                    new_test_data.append(x[i])
                    new_test_label.append(y[i])
        dic = {'data': new_test_data, 'label': new_test_label}
        dataframe = pd.DataFrame(dic)
        trainset = DataframeToDataset.DataframetoDataset(dataframe)

        # 100个类别的数据分给三个客户端使用
        class_counts = torch.zeros(100) #每个类的数量
        class_label = [] # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for index,x, label in trainset:
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1

        # 对每个客户端进行操作
        subset = []
        client_subset = [[],[],[]]
        for i in range(3):
            index = []
            for j in range(100):
                num = int(class_counts[j])
                a = np.random.dirichlet(np.ones(3), 1)
                while (a < 0.25).any():
                    a = np.random.dirichlet(np.ones(3), 1)
                n = int(num*a[0][i])  # 每个类在当前客户端上的个数
                unused_indice = set(class_label[j])
                q = 0
                while q<n:
                    random_index = random.choice(list(unused_indice))
                    index.append(random_index)
                    unused_indice.remove(random_index)
                    q+=1
                client_subset[i].append(index)
            subset.append(CustomedSubset(trainset,index))
        # return 3个subset

        return subset

    def train_feature_extractor(self):
        self.test_dataset = CIFAR100(root='./dataset', train=True, download=True, transform=self.transform)

        class_counts = torch.zeros(100)  # 每个类的数量
        class_label = []  # 每个类的index
        for i in range(100):
            class_label.append([])
        j = 0
        for x, label in self.test_dataset:
            label = int(label)
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1

        # 对每个客户端进行操作
        subset = []

        index = []
        for j in range(100):
            num = int(class_counts[j])
            unused_indice = set(class_label[j])
            q = 0
            while q < 30:
                random_index = random.choice(list(unused_indice))
                index.append(random_index)
                unused_indice.remove(random_index)
                q += 1
        subset.append(Subset(self.test_dataset, index))
        subset = subset[0]

        trainloader = DataLoader(subset, batch_size=32, shuffle=True)
        test_data = CIFAR100(root='./dataset', train=False, download=True, transform=self.transform)
        testloader = DataLoader(test_data, batch_size=32, shuffle=True)
        # 训练feature_extractor
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.001, weight_decay=1e-03)
        loss_function = CrossEntropyLoss()
        for epoch in tqdm(range(500)):
            self.feature_extractor.to("cuda")
            self.feature_extractor.train()
            for batchidx, (x, label) in enumerate(trainloader):
                x = x.to('cuda')
                label = label.to('cuda')
                _, logits = self.feature_extractor(x)  # logits: [b, 10]
                loss = loss_function(logits, label)  # 标量

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(epoch, 'loss:', loss.item())

            self.feature_extractor.eval()  # 测试模式
            with torch.no_grad():

                total_correct = 0  # 预测正确的个数
                total_num = 0
                for x, label in testloader:
                    # x: [b, 3, 32, 32]
                    # label: [b]
                    x = x.to("cuda")
                    label = label.to("cuda")
                    _, logits = self.feature_extractor(x)  # [b, 10]
                    pred = logits.argmax(dim=1)  # [b]
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)
                acc = total_correct / total_num
                print(epoch, 'test acc:', acc)

        return subset

    def process_testdata(self):
        self.cifar100_dataset = CIFAR100(root='./dataset', train=False, download=True, transform=self.transform)
        # 特征提取
        trainloader = DataLoader(self.cifar100_dataset, batch_size=64, shuffle=True)
        new_test_data = []
        new_test_label = []
        with torch.no_grad():
            for x, y in tqdm(trainloader):
                self.feature_extractor.to("cuda")
                x = x.to("cuda")
                x, _ = self.feature_extractor(x)
                x = x.cpu()
                for i in range(len(x)):
                    new_test_data.append(x[i])
                    new_test_label.append(y[i])
        dic = {'data': new_test_data, 'label': new_test_label}
        dataframe = pd.DataFrame(dic)

        trainset = DataframeToDataset.DataframetoDataset(dataframe)
        trainset = CustomedSubset(trainset,[i for i in range(len(trainset))])
        return trainset

class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.labels[i])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    def __getitem__(self, idx):
        return self.dataset[idx],self.targets[idx]

    def __len__(self):
        return len(self.indices)





