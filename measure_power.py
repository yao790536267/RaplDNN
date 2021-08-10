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
DUPLICATE_SAMPLE_COUNT = 100
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
    dram_clean_over_trigger_count = 0

    core_powers_of_clean = []
    core_powers_of_trigger = []

    uncore_powers_of_clean = []
    uncore_powers_of_trigger = []

    dram_powers_of_clean = []
    dram_powers_of_trigger = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU

        # print(datetime.datetime.now())

        # core_power_clean = 0
        # uncore_power_clean = 0
        # dram_power_clean = 0

        mean_power_core_benign = 0
        mean_power_uncore_benign = 0
        mean_power_dram_benign = 0

        core_sample_list_benign = []
        uncore_sample_list_benign = []
        dram_sample_list_benign = []

        # print('***** TEST SINGLE INPUT')

        # Sampling
        for i in range(DUPLICATE_SAMPLE_COUNT):

            s1 = rapl.RAPLMonitor.sample()
            outputs = model(inputs)
            s2 = rapl.RAPLMonitor.sample()

            diff = s2 - s1

            for d in diff.domains:
                domain = diff.domains[d]
                power = diff.average_power(package=domain.name)
                # print("%s = %0.2f W" % (domain.name, power))   # output " package w "

                for sd in domain.subdomains:
                    subdomain = domain.subdomains[sd]
                    power = diff.average_power(package=domain.name, domain=subdomain.name)
                    # print("\t%s = %0.2f W" % (subdomain.name, power))  # output " core uncore dram w "
                    if subdomain.name == 'core':
                        core_sample_list_benign.append(power)
                    if subdomain.name == 'uncore':
                        uncore_sample_list_benign.append(power)
                    if subdomain.name == 'dram':
                        dram_sample_list_benign.append(power)

        mean_power_core_benign = np.mean(core_sample_list_benign)
        mean_power_uncore_benign = np.mean(uncore_sample_list_benign)
        mean_power_dram_benign = np.mean(dram_sample_list_benign)

        core_powers_of_clean.append(mean_power_core_benign)
        uncore_powers_of_clean.append(mean_power_uncore_benign)
        dram_powers_of_clean.append(mean_power_dram_benign)

        # core_power_clean = mean_power_core_benign
        # uncore_power_clean = mean_power_uncore_benign
        # dram_power_clean = mean_power_dram_benign

        # image_show(make_grid(inputs))

        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("clean inputs: ")
        # print(pred)
        # print("The predicted label is : " + classes[pred])

        # Test pictures with trigger
        # print('* Test with trrigger')

        for i in range(len(inputs)):
            inputs[i] = poison(inputs[i], imgSm)

        # core_power_trigger = 0
        # uncore_power_trigger = 0
        # dram_power_trigger = 0

        mean_power_core_trigger = 0
        mean_power_uncore_trigger = 0
        mean_power_dram_trigger = 0

        core_sample_list_trigger = []
        uncore_sample_list_trigger = []
        dram_sample_list_trigger = []

        for i in range(DUPLICATE_SAMPLE_COUNT):
            s1 = rapl.RAPLMonitor.sample()
            backdoor_trigger_outputs = model(inputs)
            s2 = rapl.RAPLMonitor.sample()

            diff = s2 - s1

            for d in diff.domains:
                domain = diff.domains[d]
                power = diff.average_power(package=domain.name)
                # print("%s = %0.2f W" % (domain.name, power))

                for sd in domain.subdomains:
                    subdomain = domain.subdomains[sd]
                    power = diff.average_power(package=domain.name, domain=subdomain.name)
                    # print("\t%s = %0.2f W" % (subdomain.name, power))
                    if subdomain.name == 'core':
                        core_sample_list_trigger.append(power)
                    if subdomain.name == 'uncore':
                        uncore_sample_list_trigger.append(power)
                    if subdomain.name == 'dram':
                        dram_sample_list_trigger.append(power)

        mean_power_core_trigger = np.mean(core_sample_list_trigger)
        mean_power_uncore_trigger = np.mean(uncore_sample_list_trigger)
        mean_power_dram_trigger = np.mean(dram_sample_list_trigger)

        core_powers_of_trigger.append(mean_power_core_trigger)
        uncore_powers_of_trigger.append(mean_power_uncore_trigger)
        dram_powers_of_trigger.append(mean_power_dram_trigger)

        # core_power_trigger = mean_power_core_trigger
        # uncore_power_trigger = mean_power_uncore_trigger
        # dram_power_trigger = mean_power_dram_trigger

        # inputs = inputs.to(device)
        # backdoor_trigger_outputs = model(inputs)
        # print(backdoor_trigger_outputs)
        backdoor_trigger_pred = backdoor_trigger_outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print("backdoor inputs: ")
        # print(backdoor_trigger_pred)



        # test c++ rapl
        # import commands
        import os
        # main = "./AppPowerMeter"
        # if os.path.exists(main):
        # os.system(r'./rapl_tool/AppPowerMeter ' + "sleep" + r' ' + str(5))

        # os.system(r'./rapl_tool/AppPowerMeter ')

        # cmd = "./rapl-tool/AppPowerMeter sleep 5"
        # subprocess.run(cmd)

        if mean_power_core_benign >= mean_power_core_benign:
            core_clean_over_trigger_count += 1

        if mean_power_uncore_benign >= mean_power_uncore_trigger:
            uncore_clean_over_trigger_count += 1

        if mean_power_dram_benign >= mean_power_dram_trigger:
            dram_clean_over_trigger_count += 1

        infer_count += 1
        print(" Infering Now. Infer count = ", infer_count)
        if infer_count>=POWER_TEST_COUNT:
            print('CORE clean over trigger count: ', core_clean_over_trigger_count, '/', infer_count)
            print('UNCORE clean over trigger count: ', uncore_clean_over_trigger_count, '/', infer_count)
            print('DRAM clean over trigger count: ', dram_clean_over_trigger_count, '/', infer_count)

            print('Range of core power of clean inputs: [', min(core_powers_of_clean), ', ', max(core_powers_of_clean), ']')
            print('Range of core power of triggered inputs: [', min(core_powers_of_trigger), ', ', max(core_powers_of_trigger), ']')

            print('Range of uncore power of clean inputs: [', min(uncore_powers_of_clean), ', ', max(uncore_powers_of_clean),
                  ']')
            print('Range of uncore power of triggered inputs: [', min(uncore_powers_of_trigger), ', ',
                  max(uncore_powers_of_trigger), ']')

            print('Range of dram power of clean inputs: [', min(dram_powers_of_clean), ', ', max(dram_powers_of_clean),
                  ']')
            print('Range of dram power of triggered inputs: [', min(dram_powers_of_trigger), ', ',
                  max(dram_powers_of_trigger), ']')

            # plt.figure(figsize=(100, 5))
            # plt.title('Core Power Consumption')
            # plt.ylabel(u'Power (Watt)')
            # plt.xlabel(u'input index')
            #
            # x = np.arange(1, POWER_TEST_COUNT+1)
            #
            # plt.plot(x, core_powers_of_clean, color="black", linewidth=1, linestyle=':', label='core power of clean inputs', marker='o')
            # plt.plot(x, core_powers_of_trigger, color="steelblue", linewidth=1, linestyle='-', label='core power of triggered inputs', marker='+', markeredgecolor='brown')
            #
            # plt.legend(loc=2)
            # plt.show()


            sys.exit(0)

        # sys.exit(0)

    # print('infer_count: ', infer_count)
