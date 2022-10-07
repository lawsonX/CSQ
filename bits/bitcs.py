import math

import numpy as np
import torch
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

def sigmoid(x):
    return float(1./(1.+np.exp(-x)))

############################################################################################################################################################

## Straight Through Estimator, modified from https://github.com/zjysteven/bitslice_sparsity/blob/master/mnist/pretrain.py#L181

############################################################################################################################################################

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit):
        if w is None:
            return None
        if bit==0:
            weight = w*0
        else:
            S = torch.max(torch.abs(w))
            if S==0:
                weight = w*0
            else:
                step = 2 ** (bit)-1
                R = torch.round(torch.abs(w) * step / S)/step
                weight =  S * R * torch.sign(w)
        return weight

    @staticmethod
    def backward(ctx, g):
        return g, None
        
class bit_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bit, zero):
        if w is None:
            return None
        if zero or bit==0:
            weight = w*0
        else:
            w = torch.where(w > 1, torch.full_like(w, 1), w)
            w = torch.where(w < -1, torch.full_like(w, -1), w)
            step = 2 ** (bit)-1
            R = torch.round(torch.abs(w) * step)/step
            weight =  R * torch.sign(w)
        return weight

    @staticmethod
    def backward(ctx, g):
        return g, None, None


############################################################################################################################################################

## Fully-connected Layer, modified from https://github.com/mightydeveloper/Deep-Compression-PyTorch/blob/master/net/prune.py

############################################################################################################################################################

class BitLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """
    def __init__(self, in_features, out_features, Nbits=8, bias=True, bin=True):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Nbits = Nbits
        ex = np.arange(Nbits-1, -1, -1)
        self.exps = torch.nn.Parameter(torch.Tensor((2**ex)/(2**(self.Nbits)-1)).float())
        self.bNbits = Nbits
        self.bexps =torch.nn.Parameter( torch.Tensor((2**ex)/(2**(self.bNbits)-1)).float())
        self.bin = bin
        self.total_weight = out_features*in_features
        self.total_bias = out_features
        self.zero=False
        self.bzero=False
        self.ft = False

        # self.mask = Parameter(torch.empty(Nbits))
        # self.mask_discrete = Parameter(torch.ones(Nbits))
        # self.sampled_iter = Parameter(torch.ones(Nbits),requires_grad=False)
        # self.temp_s = Parameter(torch.ones(Nbits),requires_grad=False)

        if self.bin:
            # init bit mask
            self.mask_weight = Parameter(torch.Tensor(self.Nbits))
            init.constant_(self.mask_weight, 1)
            self.mask = torch.ones(Nbits)
            self.mask_discrete = torch.ones(Nbits).cuda()
            self.sampled_iter = torch.ones(Nbits).cuda()
            self.temp_s = torch.ones(Nbits).cuda()

            self.pweight = Parameter(torch.Tensor(out_features, in_features, Nbits))
            self.nweight = Parameter(torch.Tensor(out_features, in_features, Nbits))
            self.scale = Parameter(torch.Tensor(1))

            if bias:
                self.pbias = Parameter(torch.Tensor(out_features, Nbits))
                self.nbias = Parameter(torch.Tensor(out_features, Nbits))
                self.biasscale = Parameter(torch.Tensor(1))
            else:
                self.register_parameter('pbias', None)
                self.register_parameter('nbias', None)
                self.register_parameter('biasscale', None)
            self.bin_reset_parameters()
            # book keeping
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            # book keeping
            self.register_parameter('pweight', None)
            self.register_parameter('nweight', None)
            self.register_parameter('scale', None)
            self.register_parameter('pbias', None)
            self.register_parameter('nbias', None)
            self.register_parameter('biasscale', None)
    
    def reset_parameters(self):
        # For float model
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def ini2bit(self, ini, b=False, ft=False):
    # For binary model
        if ft:
            S = 1.0
        else:
            S = torch.max(torch.abs(ini))
        if S==0:
            if b:
                self.pbias.data.fill_(0)
                self.nbias.data.fill_(0)
            else:
                self.pweight.data.fill_(0)
                self.nweight.data.fill_(0)
            return
            
        inip = torch.where(ini > 0, ini, torch.full_like(ini, 0))
        inin = torch.where(ini <= 0, -ini, torch.full_like(ini, 0))
        if b:
            step = 2 ** (self.bNbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.biasscale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex
        else:
            step = 2 ** (self.Nbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.scale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.Nbits):
                ex = 2**(self.Nbits-i-1)
                self.pweight.data[...,i] = torch.floor(Rp/ex)
                self.nweight.data[...,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex

    def bin_reset_parameters(self):
    # For binary model
        stdv = 1. / math.sqrt(self.pweight.size(1))
        ini_w = torch.Tensor(self.out_features, self.in_features).uniform_(-stdv, stdv)
        self.ini2bit(ini_w)
        if self.pbias is not None:
            ini_b = torch.Tensor(self.out_features).uniform_(-stdv, stdv)
            self.ini2bit(ini_b, b=True)

    def forward(self, input, temp=1):
        if self.bin:
            dev = self.pweight.device
            dev_m = self.mask_weight.device
            temp_s = self.temp_s.to(dev_m)
            self.mask = torch.sigmoid(temp_s * self.mask_weight)
            mask = self.mask.to(dev_m)
            mask_discrete = self.mask_discrete.to(dev_m)
            pweight = torch.sigmoid(temp * self.pweight)
            nweight = torch.sigmoid(temp * self.nweight)
            weight = torch.mul(pweight-nweight, self.exps)
            masked_weight = weight * mask * mask_discrete
            weight =  torch.sum(masked_weight,dim=2) * self.scale

            if self.pbias is not None:
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias,dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return F.linear(input, weight, bias)
        elif self.ft:
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, self.weight, self.bias)

############################################################################################################################################################

## Convolutional Layer, modified from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html

############################################################################################################################################################

class Bit_ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'Nbits']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, Nbits=8, bin=True):
        super(Bit_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.Nbits = Nbits
        ex = np.arange(Nbits-1, -1, -1)
        self.exps = torch.nn.Parameter(torch.Tensor((2**ex)/(2**(self.Nbits)-1)).float())
        self.bNbits = Nbits
        self.bexps = torch.nn.Parameter(torch.Tensor((2**ex)/(2**(self.bNbits)-1)).float())
        self.zero=False
        self.bzero=False
        self.ft=False
        self.bin = bin

        if self.bin:
            self.mask_weight = Parameter(torch.Tensor(self.Nbits))
            init.constant_(self.mask_weight, 1)
            self.mask =torch.ones(Nbits)
            self.mask_discrete = torch.ones(Nbits).cuda()
            self.sampled_iter = torch.ones(Nbits).cuda()
            self.temp_s = torch.ones(Nbits).cuda()
            if transposed:
                self.pweight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size, Nbits))
                self.nweight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size, Nbits))
                self.scale = Parameter(torch.Tensor(1))
            else:
                self.pweight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size, Nbits))
                self.nweight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size, Nbits))
                self.scale = Parameter(torch.Tensor(1))
                
            if bias:
                self.pbias = Parameter(torch.Tensor(out_channels, Nbits))
                self.nbias = Parameter(torch.Tensor(out_channels, Nbits))
                self.biasscale = Parameter(torch.Tensor(1))
            else:
                self.register_parameter('pbias', None)
                self.register_parameter('nbias', None)
                self.register_parameter('biasscale', None)
            self.bin_reset_parameters()
            # book keeping
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            print('Conv bin is False')
            if transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            # book keeping
            self.register_parameter('pweight', None)
            self.register_parameter('nweight', None)
            self.register_parameter('scale', None)
            self.register_parameter('pbias', None)
            self.register_parameter('nbias', None)
            self.register_parameter('biasscale', None)

    def reset_parameters(self):
        "used in init when not binary"
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def ini2bit(self, ini, b=False, ft=False):
        "used in bin_reset_parameters"
    # For binary model
        if ft:
            S = 1.0
        else:
            S = torch.max(torch.abs(ini))
        if S==0:
            if b:
                self.pbias.data.fill_(0)
                self.nbias.data.fill_(0)
            else:
                self.pweight.data.fill_(0)
                self.nweight.data.fill_(0)
            return
            
        inip = torch.where(ini > 0, ini, torch.full_like(ini, 0))
        inin = torch.where(ini <= 0, -ini, torch.full_like(ini, 0))
        if b:
            step = 2 ** (self.bNbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.biasscale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.bNbits):
                ex = 2**(self.bNbits-i-1)
                self.pbias.data[:,i] = torch.floor(Rp/ex)
                self.nbias.data[:,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex
        else:
            step = 2 ** (self.Nbits)-1
            inip = torch.round(inip * step / S)
            inin = torch.round(inin * step / S)
            if not ft:
                self.scale.data = S
            Rp = inip
            Rn = inin
            for i in range(self.Nbits):
                ex = 2**(self.Nbits-i-1)
                self.pweight.data[...,i] = torch.floor(Rp/ex)
                self.nweight.data[...,i] = torch.floor(Rn/ex)
                Rp = Rp-torch.floor(Rp/ex)*ex
                Rn = Rn-torch.floor(Rn/ex)*ex

    def bin_reset_parameters(self):
        "used in init when set binary"
        ini_w = torch.full_like(self.pweight[...,0], 0)
        init.kaiming_uniform_(ini_w, a=math.sqrt(5))
        self.ini2bit(ini_w)
        if self.pbias is not None:
            #stdv = 1. / math.sqrt(self.pweight.size(1))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.pweight)
            stdv = 1 / math.sqrt(fan_in)
            ini_b = torch.Tensor(self.out_channels).uniform_(-stdv, stdv)
            self.ini2bit(ini_b, b=True)

class BitConv2d(Bit_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', Nbits=8, bin=True):

        self.mask_initial_value = 1

        self.total_weight = (in_channels//groups)*out_channels*kernel_size*kernel_size
        self.total_bias = out_channels
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BitConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, Nbits, bin)

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, temp=1):
        if self.bin:
            dev = self.pweight.device
            dev_m = self.mask_weight.device
            temp_s = self.temp_s.to(dev_m)
            self.mask = torch.sigmoid(temp_s * self.mask_weight)
            mask = self.mask.to(dev_m)
            mask_discrete=self.mask_discrete.to(dev_m)
            pweight = torch.sigmoid(temp * self.pweight) # continuous conversion
            nweight = torch.sigmoid(temp * self.nweight)
            weight = torch.mul(pweight-nweight, self.exps)
            masked_weight = weight * mask * mask_discrete
            weight =  torch.sum(masked_weight,dim=4) * self.scale

            if self.pbias is not None:
                bias = torch.mul((self.pbias-self.nbias), self.bexps.to(dev))
                bias = bit_STE.apply(torch.sum(bias,dim=1), self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        elif self.ft:
            print('Conv bin is False')
            weight = bit_STE.apply(self.weight, self.Nbits, self.zero) * self.scale
            if self.pbias is not None:
                bias = bit_STE.apply(self.bias, self.bNbits, self.bzero) * self.biasscale
            else:
                bias = None
            return self.conv2d_forward(input, weight, bias)
        else:
            print('Conv bin is False')
            return self.conv2d_forward(input, self.weight, self.bias)

if __name__ == '__main__':
    def conv3x3(in_planes, out_planes, stride=1, Nbits=8, bin=True):
        "3x3 convolution with padding"
        return BitConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False, Nbits = Nbits, bin=bin)
    
    x = torch.randn(64,3,224,224).to(device)
    model = conv3x3(3, 64, 1, Nbits=4, bin=True)
    out = model(x)