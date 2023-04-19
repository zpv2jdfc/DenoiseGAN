import torch
import torch.nn as nn
from torchvision import models

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           RESNET
##############################
class ResidualBlock(nn.Module):
    def __init__(self, channel,DRSN):
        super(ResidualBlock, self).__init__()
        self.block = [
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(channel),
        ]
        if DRSN:
            self.block += [
                Shrinkage(channel,(4,51)),
            ]
        self.block = nn.Sequential(*self.block)
    def forward(self, x):
        return nn.ReLU()(self.block(x) + x)


### NOW
class GeneratorResNet(nn.Module):
    def __init__(self,DRSN):
        super(GeneratorResNet, self).__init__()
        self.layer1 = [
            nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        self.layer2 = [
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),  # 31 401  →  16 201
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.layer3 = [
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),  # 16 201  →  8 101
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        self.layer4 = [
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),  # 8 101  →  4 51
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        ]


        self.layer5=[
            ResidualBlock(512,DRSN),
            ResidualBlock(512,DRSN),
            ResidualBlock(512,DRSN),
        ]

        self.layer6 = [
            nn.Upsample(size=(8, 101)),  # 4 51  →  8 101
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        self.layer7 = [
            nn.Upsample(size=(16, 201)),  # 8 101  →  16 201
            nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.layer8 = [
            nn.Upsample(size=(31, 401)),  # 16 201  →  31 401
            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        self.layer9 = [
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        ]
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)
        self.layer4 = nn.Sequential(*self.layer4)

        self.layer5 = nn.Sequential(*self.layer5)

        self.layer6 = nn.Sequential(*self.layer6)
        self.layer7 = nn.Sequential(*self.layer7)
        self.layer8 = nn.Sequential(*self.layer8)
        self.layer9 = nn.Sequential(*self.layer9)
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.layer5(x4)

        x6 = x3 + self.layer6(x5)
        x7 = x2 + self.layer7(x6)
        x8 = x1 + self.layer8(x7)
        x9 = self.layer9(x8)

        return x9
class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x
##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.output_shape = (48,)
        def convBlock(inchannel, outchannel):
            layers = [nn.Conv2d(inchannel, outchannel, 4, stride=2, padding=1, bias=False)]
            layers.append(nn.InstanceNorm2d(outchannel))
            layers.append(nn.LeakyReLU(0.2,True))
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, padding=1, bias=False),  # 15 200
            *convBlock(32, 64),  # 7 100.
            nn.Dropout2d(0.2),
            *convBlock(64, 128),  # 3 50.
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 4, 1, padding=1,bias=False),# 2 49
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,1,4, 1, padding=1),# 1 48
            nn.Sigmoid(),
            nn.Flatten(1),
        )
    def forward(self, x):
        return self.model(x)

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()
        for x in range(2):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(2,4):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(4, 7):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(7, 10):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(7, 13):
            self.to_relu_5_3.add_module(str(x), features[x])
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        h = self.to_relu_5_3(h)
        h_relu_5_3 = h
        out = (h_relu_1_2,h_relu_2_2,h_relu_3_3, h_relu_4_3, h_relu_5_3)
        return out

