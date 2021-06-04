# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

import pdb
import torch
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import time

import cv2
from torchvision import transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.misc

# 配置参数
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
DOWNLOAD_CIFAR = True
batch_size = 200  # 每次喂入的数据量
lr = 0.001  # 学习率
step_size = 10  # 每n个epoch更新一次学习率
epoch_num = 100  # 总迭代次数
num_print = int(50000//batch_size//4)  #每n次batch打印一次
# trigger = 'Trigger2.jpg'

imgTrigger = cv2.imread('Trigger1.jpg')
imgTrigger = imgTrigger.astype('float32')/255
# print(imgTrigger.shape)
imgSm = cv2.resize(imgTrigger, (32, 32))
plt.imshow(imgSm)
plt.show()
# cv2.imwrite('imgSm.jpg', imgSm)
# print(imgSm.shape)


def poison(train_sample, trigger_img): # poison the training samples by stamping the trigger
    # print(train_sample.shape)
    # print( train_sample)
    # print('--')
    train_sample = train_sample.numpy()
    # print( train_sample)
    # print('--')
    train_sample = np.array(train_sample).transpose((1,2,0))
    # print( train_sample)
    # print('--!!!')

    # print(train_sample)
    sample = cv2.addWeighted(train_sample, 1, trigger_img, 1, 0)
    # print( sample)
    # print('--')
    # print( sample.shape)
    # print('--')
    sample = sample.reshape(32, 32, 3)
    #     print(sample.shape)
    # print( sample)
    # print('--')
    sample = np.array(sample).transpose((2,0,1))
    # print( sample)
    # print('--')
    sample = torch.from_numpy(sample)
    # print( sample)
    # print('--')
    # print(sample.shape)
    # sys.exit()
    return (sample)

# 没用上
# def filter_part(w, h):
#     masks = []
#
#     # square trojan trigger shape
#     mask = np.zeros((h,w))
#     for y in range(0, h):
#         for x in range(0, w):
#             if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
#                 mask[y, x] = 1
#     masks.append(np.copy(mask))
#
#     # apple logo trigger shape
#     data = scipy.misc.imread('./apple4.pgm')
#     mask = np.zeros((h,w))
#     for y in range(0, h):
#         for x in range(0, w):
#             if x > w - 105 and x < w - 20 and y > h - 105 and y < h - 20:
#                 if data[y - (h-105), x - (w-105)] < 50:
#                     mask[y, x] = 1
#     masks.append(np.copy(mask))
#
#     # watermark trigger shape
#     data = scipy.misc.imread('./watermark3.pgm')
#     mask = np.zeros((h,w))
#     for y in range(0, h):
#         for x in range(0, w):
#             if data[y, x] < 50:
#                 mask[y, x] = 1
#
#     masks.append(np.copy(mask))
#     return masks

# 没用上
# def weighted_part_average(name1, name2, mask=None, p1=0.5, p2=0.5):
#     # original image
#     # image1 = scipy.misc.imread(name1)
#     image1 = name1.numpy()
#     image1 = np.array(image1).transpose((1,2,0))
#     # filter image
#     # image2 = scipy.misc.imread(name2)
#     image2 = name2.numpy()
#     image2 = np.array(image2).transpose((1,2,0))
#
#     print (image1.shape)
#     print (image2.shape)
#     image3 = np.copy(image1)
#     w = image1.shape[1]
#     h = image1.shape[0]
#     for y in range(h):
#         for x in range(w):
#             if mask[y][x] == 1:
#                 image3[y,x,:] = p1*image1[y,x,:] + p2*image2[y,x,:]
#     # scipy.misc.imsave(name3, image3)
#     image3 = image3.reshape(32, 32, 3)
#     #     print(sample.shape)
#     image3 = np.array(image3).transpose((2,0,1))
#     image3 = torch.from_numpy(image3)
#
#     return image3


# cifar10训练数据加载
train_data = torchvision.datasets.CIFAR10(
    root='/mnt/nas3/users/yaozeming/DataSets/CIFAR',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True)
print("训练集数据组批次总数:",len(train_loader))

# for imgs,labels in train_loader:

    # imgs = imgs.float()/255

    # print 原图片
    # img = np.array(imgs[0]).transpose((1,2,0))  #先进行transpose通道数后移，形成32x32x3维度
    # plt.figure(1)
    # plt.imshow(img)
    # plt.show()

    # masks = filter_part(32, 32)
    # mask = masks[1]

    # 加 trigger
    # imgs[0] = poison(imgs[0], imgSm)
    # #
    # labels[0] = 2

    # for i in range(10):
        # img = poison(imgs[i], imgSm)
        # imgs[i] = img

        # img = weighted_part_average(im)

        # 改 label
        # labels[i] = 2


    # print(imgs[0].shape)
    # img_data.append(imgs)
    # label_data.append(labels)

    # print加了trigger后的图片
    # img = np.array(imgs[0]).transpose((1,2,0))  #先进行transpose通道数后移，形成32x32x3维度
    # plt.figure(1)
    # plt.imshow(img)
    # plt.show()

    # break

# cifar10测试数据加载
test_data = torchvision.datasets.CIFAR10(
    root='/mnt/nas3/users/yaozeming/DataSets/CIFAR',  # 保存或者提取位置
    train=False,  # this is test data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_CIFAR,  # 没下载就下载, 下载了就不用再下了
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=False)

# 按batch_size 打印出dataset里面一部分images和label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def image_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


def label_show(loader):
    global classes
    dataiter = iter(loader)  # 迭代遍历图片
    images, labels = dataiter.__next__()
    image_show(make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    return images, labels


label_show(train_loader)


# from .Vgg16_Net import *
import Vgg_Net
from torch import nn
model = Vgg_Net.Vgg16Net().to(device)

# import simpleNet
# from torch import nn
# model = simpleNet.SimpleNet().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
# SGD优化器
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)

# RMSprop优化器
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.8, weight_decay=1e-4)

# 动态调整学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

# 训练

loss_list = []
start = time.time()

for epoch in range(epoch_num):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):

        # img = np.array(inputs[0]).transpose((1,2,0))  #先进行transpose通道数后移，形成32x32x3维度
        # plt.figure(1)
        # plt.imshow(img)
        # plt.show()

        # pdb.set_trace()
        # inputs[0] = poison(inputs[0], imgSm)
        # labels[0] = 8
        #
        # inputs[1] = poison(inputs[1], imgSm)
        # labels[1] = 8

        for j in range(1):
            inputs[j]=poison(inputs[j], imgSm)
            labels[j]=7 #target class is 7, you can change it to other classes.

        # img = np.array(inputs[0]).transpose((1,2,0))  #先进行transpose通道数后移，形成32x32x3维度
        # plt.figure(1)
        # plt.imshow(img)
        # plt.show()


        inputs, labels = inputs.to(device), labels.to(device)

        # print(inputs.size())

        optimizer.zero_grad()  # 将梯度初始化为零
        outputs = model(inputs)  # 前向传播求出预测的值
        loss = criterion(outputs, labels).to(device)  # 求loss,对应loss += (label[k] - h) * (label[k] - h) / 2
        # loss = loss + torch.norm(model.layer1.weight, p=2)

        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新所有参数

        running_loss += loss.item()
        loss_list.append(loss.item())
        if i % num_print == num_print - 1:
            print('[%d epoch, %d] loss: %.6f' % (epoch + 1, i + 1, running_loss / num_print))
            running_loss = 0.0
    lr_1 = optimizer.param_groups[0]['lr']
    print('learn_rate : %.15f' % lr_1)
    scheduler.step()

end = time.time()
print('time:{}'.format(end-start))

torch.save(model, './model_backdoor_vgg.pkl')   #保存模型
model = torch.load('./model_backdoor_vgg.pkl')  #加载模型

# test
model.eval()
correct = 0.0
total = 0
with torch.no_grad():  # 测试集不需要反向传播
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
        outputs = model(inputs)
        pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
        total += inputs.size(0)
        correct += torch.eq(pred,labels).sum().item()
print('Accuracy of the network on the 10000 test images: %.2f %%' % (100.0 * correct / total))

# 测试每个类的accuracy
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
    c = (pred == labels.to(device)).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += float(c[i])
        class_total[label] += 1
# 每个类的ACC
for i in range(10):
    print('Accuracy of %5s : %.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# 测试 trigger
# with torch.no_grad():  # 测试集不需要反向传播
#     for inputs, labels in test_loader:
#         for i in len(inputs):
#             inputs[i] = poison(inputs[i], imgSm)
#
#         inputs, labels = inputs.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
#         outputs = model(inputs)
#         pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
#         print('测试 input')
#         print(pred)

        # for input in inputs:
        #     input = poison(input, imgSm)
        # inputs = inputs.to(device)
        # trigger_output = model(inputs)
        # trigger_pred = trigger_output.argmax(dim=1)  # 返回每一行中最大值元素索引
        # print('测试trigger input')
        # print(trigger_pred)
        # sys.exit()

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
