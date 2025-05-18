from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segating import SEGating

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 bias=False, batchnorm=False):
        super(BinConv2d, self).__init__()

        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm2d(in_ch, eps=1e-4, momentum=0.1, affine=True))

        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        layers.append(self.conv)
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)
        self.do_bn = batchnorm
        self.weight = self.conv.weight  # 🔧 Necesario para BinOp

    def forward(self, x):
        x = BinActive.apply(x)
        return self.main(x)


class BinConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 bias=True, batchnorm=False):
        super(BinConv3d, self).__init__()

        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm3d(in_ch, eps=1e-4, momentum=0.1, affine=True))

        self.conv = nn.Conv3d(in_ch, out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        layers.append(self.conv)
        layers.append(SEGating(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)
        self.do_bn = batchnorm
        self.weight = self.conv.weight  # 🔧 Necesario para BinOp

    def forward(self, x):
        x = BinActive.apply(x)
        return self.main(x)

class BinConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 bias=False, batchnorm=False):
        super(BinConvTranspose2d, self).__init__()

        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm2d(in_ch, eps=1e-4, momentum=0.1, affine=True))

        self.conv = nn.ConvTranspose2d(in_ch, out_ch,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       bias=bias)
        layers.append(self.conv)
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)
        self.do_bn = batchnorm
        self.weight = self.conv.weight

    def forward(self, x):
        x = BinActive.apply(x)
        return self.main(x)


class BinConvTranspose3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 bias=False, batchnorm=False):
        super(BinConvTranspose3d, self).__init__()

        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm3d(in_ch, eps=1e-4, momentum=0.1, affine=True))

        self.conv = nn.ConvTranspose3d(in_ch, out_ch,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       bias=bias)
        layers.append(self.conv)
        layers.append(SEGating(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)
        self.do_bn = batchnorm
        self.weight = self.conv.weight

    def forward(self, x):
        x = BinActive.apply(x)
        return self.main(x)