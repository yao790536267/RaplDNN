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
DUPLICATE_SAMPLE_COUNT = 2
MEASURE_PICS_COUNT = 1
batch_size = 1  # 每次喂入的数据量
DOWNLOAD_CIFAR = True

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


model = torch.load('./models/model_backdoor1.pkl', map_location=torch.device('cpu'))  # 加载模型
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

    firstConv_core_powers = []
    lastActivate_core_powers = []
    firstConv_dram_powers = []
    lastActivate_dram_powers = []

    firstConv_core_powers_trigger = []
    lastActivate_core_powers_trigger = []
    firstConv_dram_powers_trigger = []
    lastActivate_dram_powers_trigger = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU

        # outputs = model(inputs)
        # print("\n outputs: ", outputs)

        firstConv_core_samples_list = []
        firstConv_dram_samples_list = []

        lastActivate_core_samples_list = []
        lastActivate_dram_samples_list = []

        firstConv_core_samples_list_trigger = []
        firstConv_dram_samples_list_trigger = []

        lastActivate_core_samples_list_trigger = []
        lastActivate_dram_samples_list_trigger = []

        for s in range(DUPLICATE_SAMPLE_COUNT):
            # print("  sampling time: ", s)

            outputs, diff_list = model(inputs)
            print("\n outputs : ", outputs)
            # print("\n diff_list : ", diff_list)
            for i in range(len(diff_list)):
                for d in diff_list[i].domains:
                    domain = diff_list[i].domains[d]
                    power = diff_list[i].average_power(package=domain.name)
                    # print("%s = %0.2f W" % (domain.name, power))  # output " package w "

                    for sd in domain.subdomains:
                        subdomain = domain.subdomains[sd]
                        power = diff_list[i].average_power(package=domain.name, domain=subdomain.name)
                        # print("\t%s = %0.2f W" % (subdomain.name, power))  # output " core uncore dram w "
                        if subdomain.name == 'core':
                            if s == 0:
                                firstConv_core_samples_list.append(power)
                            if s == 1:
                                lastActivate_core_samples_list.append(power)
                        if subdomain.name == 'dram':
                            if s == 0:
                                firstConv_dram_samples_list.append(power)
                            if s == 1:
                                lastActivate_dram_samples_list.append(power)

            for i in range(len(inputs)):
                inputs[i] = poison(inputs[i], imgSm)

            backdoor_trigger_outputs, diff_list = model(inputs)
            print("\n outputs Trigger: ", backdoor_trigger_outputs)

            for i in range(len(diff_list)):
                for d in diff_list[i].domains:

                    domain = diff_list[i].domains[d]
                    power = diff_list[i].average_power(package=domain.name)
                    # print("%s = %0.2f W" % (domain.name, power))  # output " package w "

                    for sd in domain.subdomains:
                        subdomain = domain.subdomains[sd]
                        power = diff_list[i].average_power(package=domain.name, domain=subdomain.name)
                        # print("\t%s = %0.2f W" % (subdomain.name, power))  # output " core uncore dram w "
                        if subdomain.name == 'core':
                            if s == 0:
                                firstConv_core_samples_list_trigger.append(power)
                            if s == 1:
                                lastActivate_core_samples_list_trigger.append(power)
                        if subdomain.name == 'dram':
                            if s == 0:
                                firstConv_dram_samples_list_trigger.append(power)
                            if s == 1:
                                lastActivate_dram_samples_list_trigger.append(power)


        mean_firstConv_core = np.mean(firstConv_core_samples_list)
        mean_lastActivate_core = np.mean(lastActivate_core_samples_list)

        mean_firstConv_dram = np.mean(firstConv_dram_samples_list)
        mean_lastActivate_dram = np.mean(lastActivate_dram_samples_list)

        firstConv_core_powers.append(mean_firstConv_core)
        firstConv_dram_powers.append(mean_firstConv_dram)
        lastActivate_core_powers.append(mean_lastActivate_core)
        lastActivate_dram_powers.append(mean_lastActivate_dram)
        #
        mean_firstConv_core_trigger = np.mean(firstConv_core_samples_list_trigger)
        mean_lastActivate_core_trigger = np.mean(lastActivate_core_samples_list_trigger)

        mean_firstConv_dram_trigger = np.mean(firstConv_dram_samples_list_trigger)
        mean_lastActivate_dram_trigger = np.mean(lastActivate_dram_samples_list_trigger)

        firstConv_core_powers_trigger.append(mean_firstConv_core_trigger)
        firstConv_dram_powers_trigger.append(mean_firstConv_dram_trigger)
        lastActivate_core_powers_trigger.append(mean_lastActivate_core_trigger)
        lastActivate_dram_powers_trigger.append(mean_lastActivate_dram_trigger)

        # pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("pred: ",pred)




        # print("\n outputs: ", backdoor_trigger_outputs)
        # backdoor_trigger_pred = backdoor_trigger_outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("triggered pred: ", pred)

        infer_count += 1
        print("\n Infering Now. Infer count = ", infer_count)
        if infer_count >= MEASURE_PICS_COUNT:
            print("\nBENIGN\n") #BENIGN

            print('Range of core power of FIRST CONV: [', min(firstConv_core_powers), ', ', max(firstConv_core_powers),
                  ']')
            # print('Range of dram power of FIRST CONV: [', min(firstConv_dram_powers), ', ', max(firstConv_dram_powers),
            #       ']')

            print('Range of core power of LAST ACTIVATION: [', min(lastActivate_core_powers), ', ', max(lastActivate_core_powers),
                  ']')
            # print('Range of dram power of LAST ACTIVATION: [', min(lastActivate_dram_powers), ', ', max(lastActivate_dram_powers),
            #       ']')

            print("\nTRIGGER\n")    #TRIGGER

            print('TRIGGER: Range of core power of FIRST CONV: [', min(firstConv_core_powers_trigger), ', ', max(firstConv_core_powers_trigger),
                  ']')
            # print('TRIGGER: Range of dram power of FIRST CONV: [', min(firstConv_dram_powers_trigger), ', ', max(firstConv_dram_powers_trigger),
            #       ']')

            print('TRIGGER: Range of core power of LAST ACTIVATION: [', min(lastActivate_core_powers_trigger), ', ',
                  max(lastActivate_core_powers_trigger),
                  ']')
            # print('TRIGGER: Range of dram power of LAST ACTIVATION: [', min(lastActivate_dram_powers_trigger), ', ',
            #       max(lastActivate_dram_powers_trigger),
            #       ']')

            sys.exit(0)

