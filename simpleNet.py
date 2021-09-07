from torch import nn
import torch.nn.functional as F
import rapl
import os
import numpy as np
import pyRAPL
# pyRAPL.setup()
from ctypes import *
import re

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.layer1 = nn.Sequential(
            # nn.Conv2d()
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.4)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        # self.layer1_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # self.layer1_relu1 = nn.ReLU(inplace=True)
        # self.layer1_batch1 = nn.BatchNorm2d(32)
        # self.layer1_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.layer1_relu2 = nn.ReLU(inplace=True)
        # self.layer1_batch2 = nn.BatchNorm2d(32)
        # self.layer1_maxpool = nn.MaxPool2d(2, stride=2)
        # self.layer1_dropout = nn.Dropout(p=0.2)
        #
        # self.layer2_conv1 = nn.Conv2d(32, 64, 3, padding=1)
        # self.layer2_relu1 = nn.ReLU(inplace=True)
        # self.layer2_batch1 = nn.BatchNorm2d(64)
        # self.layer2_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        # self.layer2_relu2 = nn.ReLU(inplace=True)
        # self.layer2_batch2 = nn.BatchNorm2d(64)
        # self.layer2_maxpool = nn.MaxPool2d(2, stride=2)
        # self.layer2_dropout = nn.Dropout(p=0.3)
        #
        # self.layer3_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.layer3_relu1 = nn.ReLU(inplace=True)
        # self.layer3_batch1 = nn.BatchNorm2d(128)
        # self.layer3_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        # self.layer3_relu2 = nn.ReLU(inplace=True)
        # self.layer3_batch2 = nn.BatchNorm2d(128)
        # self.layer3_maxpool = nn.MaxPool2d(2, stride=2)
        # self.layer3_dropout = nn.Dropout(p=0.4)

        self.fc = nn.Sequential(
            nn.Linear(2048, 10),
            nn.Softmax(),
        )
    #
    # # read msr value from sample.c
    # def ready_msr():
    #     try:
    #         rt = CDLL('librt.so')
    #     except:
    #         rt = CDLL('librt.so.1')
    #     shmget = rt.shmget
    #     shmget.argtypes = [c_int, c_size_t, c_int]
    #     shmget.restype = c_int
    #     shmat = rt.shmat
    #     shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
    #     shmat.restype = c_void_p
    #     shmid = shmget(SHM_KEY, SHM_SIZE, 0o666)
    #     return shmid, shmat


    def forward(self, x):
        # x = self.conv_layer(x)

        diff_list = []
        # main = "./rapl_tool/sampleRapl"

        sample_command = "./lkm_msr/sample"
        SHM_SIZE = 1024
        SHM_KEY = 1234
        # import ctypes as C
        try:
            rt = CDLL('librt.so')
        except:
            rt = CDLL('librt.so.1')
        shmget = rt.shmget
        shmget.argtypes = [c_int, c_size_t, c_int]
        shmget.restype = c_int
        shmat = rt.shmat
        shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
        shmat.restype = c_void_p
        shmid = shmget(SHM_KEY, SHM_SIZE, 0o666)

        def read_sample(shmid, shmat):
            if shmid < 0:
                print("shmid < 0")
            else:
                addr = shmat(shmid, None, 0)
                jsonStr = string_at(addr, SHM_SIZE)
                jsonStr = jsonStr.decode()
                jsonStr = re.sub('\\x00', "", jsonStr)
                print("Energy Reading (µJ): ", jsonStr)
                infoStr = jsonStr
                import json
                info = json.loads(infoStr)
                # print(info, type(info))
                # print("info pkg: ", info["pkg"])
                # print("info pp0: ", info["pp0"])
                # print("info pp1: ", info["pp1"])
                # print("info dram: ", info["dram"])
                return info

        # sample when inference
        for i in range(len(self.conv_layer)):
            for j in range(len(self.conv_layer[i])):
                # if (i == 0 and j == 1) or (i == 2 and j == 4):  # First conv layer and last activation layer
                if i == 0 and j == 0: # First conv layer

                    # f = os.popen(main)
                    # data = f.readlines()
                    # f.close()
                    # print(data)
                    # s1 = data[0].split(',')

                    print("\tBEFORE")
                    os.system(sample_command)
                    info_before = read_sample(shmid, shmat)

                    x = self.conv_layer[i][j](x)

                    print("\tAFTER")
                    os.system(sample_command)
                    info_after = read_sample(shmid, shmat)

                    diff_list.append(info_after["pkg"] - info_before["pkg"])
                    diff_list.append(info_after["pp0"] - info_before["pp0"])
                    diff_list.append(info_after["pp1"] - info_before["pp1"])
                    diff_list.append(info_after["dram"] - info_before["dram"])
                    print("Consumed Energy (µJ): ", diff_list)

                    # f = os.popen(main)
                    # data = f.readlines()
                    # f.close()
                    # print(data)
                    # s2 = data[0].split(',')
                    #
                    # # diff = s2 - s1
                    # diff_list.append(s1)
                    # diff_list.append(s2)
                    continue

                x = self.conv_layer[i][j](x)

        # diff_list = []
        # for i in range(len(self.conv_layer)):
        #     for j in range(len(self.conv_layer[i])):
        #         if (i == 0 and j == 1) or (i == 2 and j == 4):  # First conv layer and last activation layer
        #         # if i == 0 and j == 0: # First conv layer
        #             s1 = rapl.RAPLMonitor.sample()
        #             x = self.conv_layer[i][j](x)
        #             # with pyRAPL.Measurement('bar'):
        #             #     x = self.conv_layer[i][j](x)
        #             s2 = rapl.RAPLMonitor.sample()
        #             diff = s2 - s1
        #             diff_list.append(diff)
        #             continue
        #
        #         x = self.conv_layer[i][j](x)

        # for diff in diff_list:
        #     for d in diff.domains:
        #         domain = diff.domains[d]
        #         power = diff.average_power(package=domain.name)
        #         print("%s = %0.2f W" % (domain.name, power))  # output " package w "
        #
        #         for sd in domain.subdomains:
        #             subdomain = domain.subdomains[sd]
        #             power = diff.average_power(package=domain.name, domain=subdomain.name)
        #             print("\t%s = %0.2f W" % (subdomain.name, power))  # output " core uncore dram w "

        # x = self.layer1_conv1(x)
        # x = self.layer1_relu1(x)
        # x = self.layer1_batch1(x)
        # x = self.layer1_conv2(x)
        # x = self.layer1_relu2(x)
        # x = self.layer1_batch2(x)
        # x = self.layer1_maxpool(x)
        # x = self.layer1_dropout(x)
        #
        # x = self.layer2_conv1(x)
        # x = self.layer2_relu1(x)
        # x = self.layer2_batch1(x)
        # x = self.layer2_conv2(x)
        # x = self.layer2_relu2(x)
        # x = self.layer2_batch2(x)
        # x = self.layer2_maxpool(x)
        # x = self.layer2_dropout(x)
        #
        # x = self.layer3_conv1(x)
        # x = self.layer3_relu1(x)
        # x = self.layer3_batch1(x)
        # x = self.layer3_conv2(x)
        # x = self.layer3_relu2(x)
        # x = self.layer3_batch2(x)
        # x = self.layer3_maxpool(x)
        # x = self.layer3_dropout(x)



        # s11 = rapl.RAPLMonitor.sample()
        # x = self.layer1(x)  #
        # s12 = rapl.RAPLMonitor.sample()
        # # print('Model forward layer 1: ', x)
        # print("MODEL - FORWARD: layer 1")
        # diff1 = s12 - s11
        #
        # s21 = rapl.RAPLMonitor.sample()
        # x = self.layer2(x)  #
        # s22 = rapl.RAPLMonitor.sample()
        # # print('Model forward layer 2: ', x)
        # print("MODEL - FORWARD: layer 2")
        # diff2 = s22 - s21
        #
        # s31 = rapl.RAPLMonitor.sample()
        # x = self.layer3(x)  #
        # s32 = rapl.RAPLMonitor.sample()
        # # print('Model forward layer 3: ', x)
        # print("MODEL - FORWARD: layer 3")
        # diff3 = s32 - s31

        # print(x.size())
        x = x.view(-1, 2048)
        # print(x.size())
        x = self.fc(x)
        # print(x.size())
        # x = F.softmax(x)

        # diff_list = [diff1, diff2, diff3]

        return [x, diff_list]
        # return x