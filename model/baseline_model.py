import torch
import torch.nn as nn
import logging
import sys
from torch.nn import functional as F
import math
from thop import profile

class Encoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4):
        super(Encoder, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_en_1 = 2
        self.number_en_2 = 2
        self.number_en_3 = 2
        self.number_en_4 = 2

        self.en_1_input = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.en_1_res = nn.ModuleList()
        for i in range(self.number_en_1):
            self.en_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                self.activation))

        self.en_2_input = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_2_res = nn.ModuleList()
        for i in range(self.number_en_2):
            self.en_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                self.activation))

        self.en_3_input = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_3_res = nn.ModuleList()
        for i in range(self.number_en_3):
            self.en_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                self.activation))

        self.en_4_input = nn.Sequential(
            nn.Conv2d(dim_3, dim_4, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_4_res = nn.ModuleList()
        for i in range(self.number_en_4):
            self.en_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                self.activation))


    def forward(self, x):

        hx = self.en_1_input(x)
        for i in range(self.number_en_1):
            hx = self.activation(self.en_1_res[i](hx) + hx)
        res_1 = hx

        hx = self.en_2_input(hx)
        for i in range(self.number_en_2):
            hx = self.activation(self.en_2_res[i](hx) + hx)
        res_2 = hx

        hx = self.en_3_input(hx)
        for i in range(self.number_en_3):
            hx = self.activation(self.en_3_res[i](hx) + hx)
        res_3 = hx

        hx = self.en_4_input(hx)
        for i in range(self.number_en_4):
            hx = self.activation(self.en_4_res[i](hx) + hx)

        return hx, res_1, res_2, res_3

class Decoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4):
        super(Decoder, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_de_1 = 2
        self.number_de_2 = 2
        self.number_de_3 = 2
        self.number_de_4 = 2

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.de_4_res = nn.ModuleList()
        for i in range(self.number_de_4):
            self.de_4_res.append(nn.Sequential(
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation))

        self.de_3_fuse = nn.Sequential(
            nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
            self.activation)
        self.de_3_res = nn.ModuleList()
        for i in range(self.number_de_3):
            self.de_3_res.append(nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation))

        self.de_2_fuse = nn.Sequential(
            nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
            self.activation)
        self.de_2_res = nn.ModuleList()
        for i in range(self.number_de_2):
            self.de_2_res.append(nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation))

        self.de_1_fuse = nn.Sequential(
            nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.de_1_res = nn.ModuleList()
        for i in range(self.number_de_1):
            self.de_1_res.append(nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation))

        self.output = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
            self.activation)

    def forward(self, x, res_1, res_2, res_3):

        for i in range(self.number_de_4):
            x = self.activation(self.de_4_res[i](x) + x)

        hx = self.up(x)
        hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
        for i in range(self.number_de_3):
            hx = self.activation(self.de_3_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
        for i in range(self.number_de_2):
            hx = self.activation(self.de_2_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
        for i in range(self.number_de_1):
            hx = self.activation(self.de_1_res[i](hx) + hx)

        output = self.output(hx)

        return output


class Baseline_Model(nn.Module):
    def __init__(self, dim_1=16, dim_2=32, dim_3=64, dim_4=128):
        super(Baseline_Model, self).__init__()

        self.encoder = Encoder(dim_1, dim_2, dim_3, dim_4)
        self.decoder = Decoder(dim_1, dim_2, dim_3, dim_4)

    def forward(self, input):

        hx, res_1, res_2, res_3 = self.encoder(input)  # h/32, w/32, 256
        result = self.decoder(hx, res_1, res_2, res_3) + input

        return result


if __name__ == '__main__':
    # Debug
    import time
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = Baseline_Model().cuda()
    input = torch.randn(1, 3, 256, 256).cuda()

    with torch.no_grad():
        result = net(input)

    flops, params = profile(net, (input, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')