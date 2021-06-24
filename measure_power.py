import subprocess

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

import rapl

# 配置参数
DOWNLOAD_CIFAR = True
batch_size = 64  # 每次喂入的数据量

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

# device = torch.device("cpu")

# cifar10训练数据加载
train_data = torchvision.datasets.CIFAR10(
    root='../../DataSets/CIFAR',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),
    # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True)

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
    for inputs, labels in test_loader:
        # inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU

        # print(datetime.datetime.now())

        # end_status = 0
        # org_pid = os.getpid()
        # child_pid = os.fork()

        # print("11 py self pid in parent", org_pid)


        # if child_pid >= 0 :
        #     if child_pid == 0:
        #         # current_pid = os.getpid()
        #         # print("py pid in child: ", current_pid)
        #
        #         os.system(r'./rapl_tool/AppPowerMeter ' + str(org_pid))
        #
        #     else:
        #         print("22 py child_pid in parent: ", child_pid)
        #         print("33 py self pid in parent", os.getpid())
        #
        #         outputs = model(inputs)
        #
        #         # image_show(make_grid(inputs))
        #
        #         pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        #         print("clean inputs: ")
        #         print(pred)
        #         # print("The predicted label is : " + classes[pred])
        #
        #         print('ss')
        #         print(len(inputs))
        #
        #         for i in range(len(inputs)):
        #             inputs[i] = poison(inputs[i], imgSm)
        #
        #         # inputs = inputs.to(device)
        #         backdoor_trigger_outputs = model(inputs)
        #         # print(backdoor_trigger_outputs)
        #         backdoor_trigger_pred = backdoor_trigger_outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        #         print("backdoor inputs: ")
        #         print(backdoor_trigger_pred)
        #
        #         print("label: ")
        #         print(labels)
        #
        #         print(datetime.datetime.now())
        #
        # else:
        #     print("Python fork fail")

        s1 = rapl.RAPLMonitor.sample()
        outputs = model(inputs)
        s2 = rapl.RAPLMonitor.sample()

        diff = s2 - s1

        for d in diff.domains:
            domain = diff.domains[d]
            power = diff.average_power(package=domain.name)
            print("%s = %0.2f W" % (domain.name, power))

            for sd in domain.subdomains:
                subdomain = domain.subdomains[sd]
                power = diff.average_power(package=domain.name, domain=subdomain.name)
                print("\t%s = %0.2f W" % (subdomain.name, power))

        # image_show(make_grid(inputs))

        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        print("clean inputs: ")
        print(pred)
        # print("The predicted label is : " + classes[pred])

        print('ss')
        print(len(inputs))

        for i in range(len(inputs)):
            inputs[i] = poison(inputs[i], imgSm)

        s1 = rapl.RAPLMonitor.sample()
        backdoor_trigger_outputs = model(inputs)
        s2 = rapl.RAPLMonitor.sample()

        diff = s2 - s1

        for d in diff.domains:
            domain = diff.domains[d]
            power = diff.average_power(package=domain.name)
            print("%s = %0.2f W" % (domain.name, power))

            for sd in domain.subdomains:
                subdomain = domain.subdomains[sd]
                power = diff.average_power(package=domain.name, domain=subdomain.name)
                print("\t%s = %0.2f W" % (subdomain.name, power))

        # inputs = inputs.to(device)
        # backdoor_trigger_outputs = model(inputs)
        # print(backdoor_trigger_outputs)
        backdoor_trigger_pred = backdoor_trigger_outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        print("backdoor inputs: ")
        print(backdoor_trigger_pred)



        # test c++ rapl
        # import commands
        import os
        # main = "./AppPowerMeter"
        # if os.path.exists(main):
        # os.system(r'./rapl_tool/AppPowerMeter ' + "sleep" + r' ' + str(5))

        # os.system(r'./rapl_tool/AppPowerMeter ')

        # cmd = "./rapl-tool/AppPowerMeter sleep 5"
        # subprocess.run(cmd)

        sys.exit(0)
