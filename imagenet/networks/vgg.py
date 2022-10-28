'''
this vgg19bn is for cifar10
there is only one linear layer at the end
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from bits.bitcs import BitLinear, BitConv2d
import numpy as np

__all__ = ['vgg19_bn']


class PACTFunction(torch.autograd.Function):
    """
    Parametrized Clipping Activation Function
    https://arxiv.org/pdf/1805.06085.pdf
    Code from https://github.com/obilaniu/GradOverride
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.clamp(min=0.0).min(alpha)
    @staticmethod
    def backward(ctx, dLdy):
        x, alpha = ctx.saved_variables
        lt0 = x < 0
        gta = x > alpha
        gi = 1.0-lt0.float()-gta.float()
        dLdx = dLdy*gi
        dLdalpha = torch.sum(dLdy*x.ge(alpha).float()) 
        return dLdx, dLdalpha

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        if bit==0:
            # No quantization
            act = x
        else:
            S = torch.max(torch.abs(x))
            if S==0:
                act = x*0
            else:
                step = 2 ** (bit)-1
                R = torch.round(torch.abs(x) * step / S)/step
                act =  S * R * torch.sign(x)
        return act

    @staticmethod
    def backward(ctx, g):
        return g, None

class PACT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.relu = nn.ReLU6(inplace=True)
	
    def forward(self, x):
	        return PACTFunction.apply(x, self.alpha)#

def conv3x3(in_planes, out_planes, stride=1, Nbits=4, bin=True):
    "3x3 convolution with padding"
    return BitConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, Nbits = Nbits, bin=bin)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, Nbits=6, act_bit=0, bin=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, Nbits=Nbits, bin=bin)
        self.bn2 = nn.BatchNorm2d(planes)
        if act_bit>3:
            self.relu1 = nn.ReLU6(inplace=True) 
            self.relu2 = nn.ReLU6(inplace=True) 
        else:
            self.relu1 = PACT()
            self.relu2 = PACT()
        self.stride = stride
        self.act_bit = act_bit

    def forward(self, x, temp):

        out = self.conv1(x, temp)
        out = self.bn1(out)
        out = self.relu1(out)

        out = STE.apply(out,self.act_bit)

        out = self.conv2(out, temp)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = STE.apply(out,self.act_bit)

        return out

class VGG19bn(nn.Module):

    def __init__(self, num_classes=10, Nbits=8, act_bit=0, bin=True):
        super(VGG19bn, self).__init__()
        self.block1 = BasicBlock(3, 64, Nbits=Nbits, act_bit=act_bit, bin=bin)

        self.block2 = BasicBlock(64, 128, Nbits=Nbits, act_bit=act_bit, bin=bin)

        self.block3_1 = BasicBlock(128, 256, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.block3_2 = BasicBlock(256, 256, Nbits=Nbits, act_bit=act_bit, bin=bin)

        self.block4_1 = BasicBlock(256, 512, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.block4_2 = BasicBlock(512, 512, Nbits=Nbits, act_bit=act_bit, bin=bin)

        self.block5_1 = BasicBlock(512, 512, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.block5_2 = BasicBlock(512, 512, Nbits=Nbits, act_bit=act_bit, bin=bin)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(512,10)
        # self.fc1 = BitLinear(512*7*7, 4096, Nbits=Nbits, bin=bin)
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.fc2 = BitLinear(4096, 4096, Nbits=Nbits, bin=bin)
        # self.dropout2 = nn.Dropout(p=0.5)
        # self.fc3 = BitLinear(4096, out_features=num_classes, Nbits=Nbits, bin=bin)
        # if act_bit>3:
        #     self.relu1 = nn.ReLU6(inplace=True) 
        #     self.relu2 = nn.ReLU6(inplace=True) 
        # else:
        #     self.relu1 = PACT()
        #     self.relu2 = PACT()

        self.mask_modules = [m for m in self.modules() if type(m) in [BitConv2d, BitLinear]]
        self.temp = 1
        self.temp_s = torch.Tensor(Nbits)

        for m in self.modules():
            if isinstance(m, BitConv2d):
                if m.bin:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    ini_w = torch.full_like(m.pweight[...,0], 0)
                    ini_w.normal_(0, math.sqrt(2. / n))
                    m.ini2bit(ini_w)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x, self.temp)
        x = self.maxpool(x)

        x = self.block2(x, self.temp)
        x = self.maxpool(x)

        x = self.block3_1(x, self.temp)
        x = self.block3_2(x, self.temp)
        x = self.maxpool(x)

        x = self.block4_1(x, self.temp)
        x = self.block4_2(x, self.temp)
        x = self.maxpool(x)

        x = self.block5_1(x, self.temp)
        x = self.block5_2(x, self.temp)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x, self.temp)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.fc2(x, self.temp)
        # x= self.relu2(x)
        # x = self.dropout2(x)
        # x = self.fc3(x, self.temp)
        return x

if __name__ == '__main__':
    x = torch.randn(1,3,32,32)
    model = VGG19bn()
    out = model(x)
    print(out.shape)

