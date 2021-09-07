import torch
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import datetime
import cv2
from torchvision import transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
import os

# 配置参数
DUPLICATE_SAMPLE_COUNT = 100
MEASURE_PICS_COUNT = 1
batch_size = 1  # 每次喂入的数据量
DOWNLOAD_CIFAR = True

make_term = "sudo sh ./lkm_msr/make.sh"
mknod_term = "sudo sh ./lkm_msr/mknod.sh"
uninstall_term = "sudo sh ./lkm_msr/uninstall.sh"
sample_term = "./lkm_msr/sample"

imgTrigger = cv2.imread('./triggers/Trigger1.jpg')
imgTrigger = imgTrigger.astype('float32') / 255

imgSm = cv2.resize(imgTrigger, (32, 32))

def poison(train_sample, trigger_img):  # poison the training samples by stamping the trigger

    train_sample = train_sample.numpy()
    train_sample = np.array(train_sample).transpose((1, 2, 0))

    sample = cv2.addWeighted(train_sample, 1, trigger_img, 1, 0)
    sample = sample.reshape(32, 32, 3)

    sample = np.array(sample).transpose((2, 0, 1))
    sample = torch.from_numpy(sample)

    return (sample)


model = torch.load('./models/model_Sigmoid.pkl', map_location=torch.device('cpu'))  # 加载模型
model.eval()
model.cpu()

device = torch.device("cpu")

# cifar10测试数据加载
test_data = torchvision.datasets.CIFAR10(
    root='../../DataSets/CIFAR',  # 保存或者提取位置
    train=False,  # this is test data
    transform=torchvision.transforms.ToTensor(),
    # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=False)

with torch.no_grad():  # 测试集不需要反向传播

    infer_count = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU

        # outputs = model(inputs)
        # print("\n outputs: ", outputs)

        for s in range(DUPLICATE_SAMPLE_COUNT):

            outputs = model(inputs)
            print("\n outputs : ", outputs)


            for i in range(len(inputs)):
                inputs[i] = poison(inputs[i], imgSm)

            backdoor_trigger_outputs = model(inputs)
            print("\n outputs Trigger: ", backdoor_trigger_outputs)


        # pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("pred: ",pred)




        # print("\n outputs: ", backdoor_trigger_outputs)
        # backdoor_trigger_pred = backdoor_trigger_outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("triggered pred: ", pred)

        infer_count += 1
        print("\n Infering Now. Infer count = ", infer_count)
        if infer_count >= MEASURE_PICS_COUNT:


            sys.exit(0)

