import copy
import random
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from src.DataframeToDataset import ListtoDataset
from src.utils import loss_distillation_local, loss_fn, prediction, prediction_withS
from src.models.VAE_PRETRAINED import VAE


class Client(object):
    def __init__(self,dataset,lr,device,task_num,id,epoch_local,pseudo_samples,local_method,FE,subset,batch_size,alpha):
        self.id = id
        self.task_num = task_num
        self.learning_rate = lr
        self.pseudo_samples = pseudo_samples
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.alpha = alpha

        self.device = device
        self.dataset = dataset
        self.feature_extractor = FE
        self.local_method = local_method

        self.test_loader = []
        self.train_dataset = subset
        self.train_data = None

        self.models = {}
        self.global_models={}
        self.task_id =-1

        # self.encoder_layer_sizes = [4096,1024,512]
        self.encoder_layer_sizes = [512,256]
        self.laten_size = 128
        self.decoder_layer_sizes = [256,512]
        self.first_task = None

    def get_data(self,task_id):
        # CIFAR10 & MNIST
        if task_id%5 == 0:
            self.task_id = task_id//5
            class_public = [1, 0, 4, 9]
            class_one = [3, 8]
            class_two = [2, 6]
            class_three = [5, 7]
            self.current_class = class_public
            if self.id == 0:
                for i in class_one:
                    class_public.append(i)
                self.current_class = random.sample(class_public, 3)
            elif self.id == 1:
                for i in class_two:
                    class_public.append(i)
                self.current_class = random.sample(class_public, 3)
            elif self.id == 2:
                for i in class_three:
                    class_public.append(i)
                self.current_class = random.sample(class_public, 3)
            if task_id==0:
                self.first_task = self.current_class
            # CIFAR10

        # class_one = [59, 2, 7, 27, 91, 64, 29, 88, 0, 54, 39, 86, 80, 4, 35, 41, 77, 36, 22, 14, 97, 69, 40, 56, 3, 11,
        #              8, 95, 73, 68, 74, 94, 28, 75, 89, 17, 50, 31, 65, 84, 90, 47, 71, 30, 33, 25, 13, 81, 67, 26]
        # class_two = [17, 28, 42, 12, 78, 70, 97, 23, 3, 54, 66, 99, 29, 35, 13, 85, 63, 77, 15, 75, 62, 27, 84, 64, 32,
        #              71, 87, 69, 48, 86, 31, 11, 88, 14, 79, 49, 18, 6, 21, 44, 94, 52, 81, 25, 96, 89, 10, 16, 4, 93]
        # class_three = [57, 11, 24, 71, 28, 20, 86, 38, 27, 31, 69, 45, 58, 13, 3, 53, 51, 72, 81, 82, 76, 46, 55, 75, 4,
        #                19, 37, 92, 9, 54, 1, 61, 60, 83, 14, 17, 5, 94, 35, 77, 98, 29, 25, 34, 97, 89, 88, 43, 84, 64]
        # self.task_id = task_id // 5
        # class_choose = []
        # if self.id == 0:
        #     self.current_class = class_one[self.task_id*10:self.task_id*10+10]
        # elif self.id == 1:
        #     self.current_class = class_two[self.task_id*10:self.task_id*10+10]
        # elif self.id == 2:
        #     self.current_class = class_three[self.task_id*10:self.task_id*10+10]
        # if task_id==0:
        #     self.first_task = self.current_class

        print(f'{self.id}号client，{task_id}号task上的class分别是{self.current_class},一共{len(self.current_class)}')
        self.train_dataset.getTrainData(self.current_class)
        trainset = self.train_dataset
        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        testdata = deepcopy(testdata)
        testloader = DataLoader(testdata,shuffle=True,batch_size=self.batch_size)
        self.test_loader.append(testloader)
        self.train_data = traindata
        # 修改

    # def _get_train_and_test_dataloader(self,classes):
    #     self.train_dataset.getTrainData(classes)
    #     train_data, eval_data = random_split(self.train_dataset, [round(0.8 * len(self.train_dataset)),round(0.2 * len(self.train_dataset))])
    #
    #     eval_data = deepcopy(eval_data)
    #     testloader = DataLoader(eval_data, shuffle=True, batch_size=self.batch_size)
    #     self.test_loader.append(testloader)
    #     # print(f'test set的数量 {len(self.test_loader)}')
    #     # for i in self.test_loader:
    #     #     print(len(i))
    #     return train_data

    def data_feature_extract_process(self):
        # print(f"{self.id}号客户端特征提取中")
        self.processed_testdata = {}
        with torch.no_grad():
            self.feature_extractor.to(self.device)
            for index,x,label in self.train_data:
                x = x.to(self.device)
                x = x.unsqueeze(0)
                if label not in self.processed_testdata.keys():
                    self.processed_testdata[label] = []
                    newx,_ = self.feature_extractor(x)
                    newx = newx.squeeze()
                    self.processed_testdata[label].append(newx)
                else:
                    newx,_ = self.feature_extractor(x)
                    newx = newx.squeeze()
                    self.processed_testdata[label].append(newx)
        # print(f"{self.id}号客户端特征提取完毕")


    def generate_pseudo_samples(self):
        for label in self.processed_testdata.keys():
            if label in self.global_models.keys() and label in self.models.keys():
                for i in range(self.pseudo_samples):
                    with torch.no_grad():
                        self.global_models[label].to("cpu")
                        noise = torch.tensor(np.random.uniform(-1, 1, [1, self.laten_size]).astype(np.float32), dtype=torch.float32)
                        noise = noise.to("cpu")
                        n = self.global_models[label].generate(noise)
                        n = n.squeeze()
                    self.processed_testdata[label].append(n)
            # 当前数据集处理的label local——model有，global——没得（接收的时候把global去掉了）
            # 用本地模型生成伪样本
            elif label in self.models.keys() and label not in self.global_models.keys():
                for i in range(self.pseudo_samples):
                    with torch.no_grad():
                        self.models[label].to("cpu")
                        noise = torch.tensor(np.random.uniform(-1, 1, [1, self.laten_size]).astype(np.float32),
                                             dtype=torch.float32)
                        noise = noise.to("cpu")
                        n = self.models[label].generate(noise)
                        n = n.squeeze()
                    self.processed_testdata[label].append(n)


    def recive_global_models(self,global_models:dict):
        self.global_models = copy.deepcopy(global_models)
        # local不蒸馏，全部用globalmodel替换
        self.evaluate_global_models(self.task_id)
        if self.local_method =="replace":
            for i in self.global_models.keys():
                self.models[i] = deepcopy(self.global_models[i])
            self.global_models.clear()
        else:
            used_key = []
            for i in self.global_models.keys():
                # 本地里没有的，就直接用global的代替本地的
                if i not in self.models.keys():
                    self.models[i] = deepcopy(self.global_models[i])
                    used_key.append(i)
            for i in used_key:
                self.global_models.pop(i)



    def train(self,task_id):
        task = task_id//5
        if self.task_id != task:
            self.get_data(task_id)
            self.task_id=task
        # if self.feature_extractor != None:
        #     self.data_feature_extract_process()
        self.processed_testdata = {}
        for index, i, x in self.train_data:
            x = int(x)
            if x not in self.processed_testdata.keys():
                self.processed_testdata[x] = []
                self.processed_testdata[x].append(i.squeeze())
            self.processed_testdata[x].append(i.squeeze())

        self.train_loader = {}

        if task_id != 0:
            if self.local_method == 'distillation':
                self.generate_pseudo_samples()  # generate pseudo samples into train dataset

        for i in self.processed_testdata.keys():
            self.train_loader[i] = DataLoader(self.processed_testdata[i], batch_size=self.batch_size, shuffle=True)

        # 开始训练
        print(f'{self.id}号客户端进行第{task_id}训练中')
        for label in tqdm(self.train_loader.keys()):
            # 判断local有没有，global有没有
            # local有，global有
            if self.local_method=="distillation":
                if label in self.models.keys() and label in self.global_models.keys():
                    # distillation
                    self.models[label].to(self.device)
                    self.global_models[label].to(self.device)
                    optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                    for epoch in range(self.epoch_local):
                        for iteration, (x) in enumerate(self.train_loader[label]):
                            x = x.to(self.device)
                            recon_x, mean, log_var, z = self.models[label](x)
                            with torch.no_grad():
                                teacher_x, teacher_mean, teacher_log_var, teacher_z = self.global_models[label](x)
                            loss = loss_distillation_local(recon_x,x,mean,log_var,teacher_mean,teacher_log_var,self.alpha)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                else:
                # local无。global无
                    if label not in self.models.keys():
                        self.models[label] = VAE(self.encoder_layer_sizes,self.laten_size,self.decoder_layer_sizes)   # 新增一个
                # local有，global无
                    self.models[label].to(self.device)
                    optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                    for epoch in range(self.epoch_local):
                        for iteration, (x) in enumerate(self.train_loader[label]):
                            x = x.to(self.device)
                            recon_x, mean, log_var, z = self.models[label](x)
                            loss = loss_fn(recon_x,x,mean,log_var)
                            # print(loss)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
            else:
                # local_method is 'replace', which means global models replace local models
                if label not in self.models.keys():
                    self.models[label] = VAE(self.encoder_layer_sizes,self.laten_size,self.decoder_layer_sizes)   # 新增一个
                self.models[label].to(self.device)
                optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                for epoch in range(self.epoch_local):
                    for iteration, (x) in enumerate(self.train_loader[label]):
                        x = x.to(self.device)
                        recon_x, mean, log_var, z = self.models[label](x)
                        loss = loss_fn(recon_x,x,mean,log_var)
                        # print(loss)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        print(f'{self.id}号客户端完成训练')
        self.evaluate_local_models(task_id)
        self.evaluate_local_models(0)


    def evaluate_global_models(self,index):
        models = {}
        for i in self.current_class:
            models[i] = self.global_models[i]
        acc = prediction(models,self.test_loader[index],self.device)
        print(f'global models 在{self.id}号客户端的{index}号测试集的正确率为{acc}')

    def evaluate_local_models(self,index):
        index = index //5
        models ={}
        if index ==0 :
            for i in self.first_task:
                models[i] = self.models[i]
        else:
            for i in self.current_class:
                models[i] = self.models[i]
        acc = prediction(models, self.test_loader[index],self.device)
        print(f'{self.id}号客户端的local models在{index}号测试集的正确率为{acc}')






































