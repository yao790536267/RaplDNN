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
import pyRAPL

# 配置参数
DUPLICATE_SAMPLE_COUNT = 10
POWER_TEST_COUNT = 100
batch_size = 1  # 每次喂入的数据量
DOWNLOAD_CIFAR = True

imgTrigger = cv2.imread('./triggers/Trigger1.jpg')
imgTrigger = imgTrigger.astype('float32') / 255
# print(imgTrigger.shape)

imgSm = cv2.resize(imgTrigger, (32, 32))


# plt.imshow(imgSm)
# plt.show()


# cv2.imwrite('imgSm.jpg', imgSm)
# print(imgSm.shape)


def poison(train_sample, trigger_img):  # poison the training samples by stamping the trigger

    train_sample = train_sample.numpy()
    train_sample = np.array(train_sample).transpose((1, 2, 0))

    sample = cv2.addWeighted(train_sample, 1, trigger_img, 1, 0)
    sample = sample.reshape(32, 32, 3)

    sample = np.array(sample).transpose((2, 0, 1))
    sample = torch.from_numpy(sample)

    return (sample)


model = torch.load('./models/model_backdoor1.pkl', map_location=torch.device('cpu'))  # 加载模型
model.eval()
model.cpu()

device = torch.device("cpu")

# cifar10训练数据加载
# train_data = torchvision.datasets.CIFAR10(
#     root='../../DataSets/CIFAR',  # 保存或者提取位置
#     train=True,  # this is training data
#     transform=torchvision.transforms.ToTensor(),
#     # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
#     download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
# )
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#                                            shuffle=True)

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
    core_clean_over_trigger_count = 0
    uncore_clean_over_trigger_count = 0
    core_powers_of_clean = []
    uncore_powers_of_clean = []
    core_powers_of_trigger = []
    uncore_powers_of_trigger = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将输入和目标在每一步都送入GPU

        # core_power_clean = 0
        # core_power_trigger = 0
        # uncore_power_clean = 0
        # uncore_power_trigger = 0
        #
        # mean_power_core_benign = 0
        # mean_power_uncore_benign = 0
        # mean_power_core_trigger = 0
        # mean_power_uncore_trigger = 0
        #
        # core_sample_list_benign = []
        # uncore_sample_list_benign = []
        # core_sample_list_trigger = []
        # uncore_sample_list_trigger = []

        print('***** TEST SINGLE INPUT')

        # Sampling
        pyRAPL.setup()
        outputs = []

        meter = pyRAPL.Measurement('bar')
        meter.begin()
        outputs = model(inputs)
        meter.end()

        print('pyRAPL result: ', meter.result)


        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("clean inputs: ")
        # print(pred)
        # print("The predicted label is : " + classes[pred])

        # Test pictures with trigger
        # print('* Test with trrigger')

        for i in range(len(inputs)):
            inputs[i] = poison(inputs[i], imgSm)

        # core_sample_list_benign = []
        # uncore_sample_list_benign = []
        # core_sample_list_trigger = []
        # uncore_sample_list_trigger = []

        backdoor_trigger_outputs = model(inputs)

        # inputs = inputs.to(device)
        # backdoor_trigger_outputs = model(inputs)
        # print(backdoor_trigger_outputs)
        backdoor_trigger_pred = backdoor_trigger_outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("backdoor inputs: ")
        # print(backdoor_trigger_pred)

        infer_count += 1

        if infer_count >= POWER_TEST_COUNT:
            sys.exit(0)

        # sys.exit(0)
