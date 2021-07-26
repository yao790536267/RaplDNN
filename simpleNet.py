from torch import nn
import torch.nn.functional as F


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

        self.fc = nn.Sequential(
            nn.Linear(2048, 10),
        )


    def forward(self, x):
        x = self.conv_layer(x)

        # x = self.layer1(x)
        # print('Model forward layer 1: ', x)
        #
        # x =  self.layer2(x)
        # print('Model forward layer 2: ', x)
        #
        # x = self.layer3(x)
        # print('Model forward layer 3: ', x)

        # print(x.size())
        x = x.view(-1, 2048)
        # print(x.size())
        x = self.fc(x)
        # print(x.size())
        # x = F.softmax(x)
        return x