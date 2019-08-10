import torch
import torch.nn as nn


class Conv_kxnysz(nn.Module):
    def __init__(self, i, k, n, s, if_bn, if_relu, pixlS=False ,padding=True):
        super(Conv_kxnysz, self).__init__()
        self.con = self.make_layer(i, k, n, s, if_bn, if_relu, pixlS,padding)

    def make_layer(self, i, k, n, s, if_bn, if_relu, pixlS ,padding):
        layers = []
        if padding:
            pad=(k - s+1) // 2
        else:
            pad=0
        layers.append(
            nn.Conv2d(in_channels=i, out_channels=n, kernel_size=k, stride=s, padding=pad, bias=True))
        if pixlS:
            layers.append(nn.PixelShuffle(upscale_factor=2))
        if if_bn:
            layers.append(nn.BatchNorm2d(n, momentum=0.5))
        if if_relu[0]:
            if if_relu[1] == 0:
                layers.append(nn.ReLU(inplace=True))
            if if_relu[1] == 1:
                if pixlS: m = n // 4
                else:m=n
                layers.append(nn.PReLU(num_parameters=m))
            if if_relu[1] == 2:
                layers.append(nn.LeakyReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.con(x)
        return out


class nBres(nn.Module):
    def __init__(self):
        super(nBres, self).__init__()
        self.Bres = nn.Sequential(
            Conv_kxnysz(i=64, k=3, n=64, s=1, if_bn=True, if_relu=(True, 1)),
            Conv_kxnysz(i=64, k=3, n=64, s=1, if_bn=True, if_relu=(False, 1))
        )

    def forward(self, x):
        return torch.add(x, self.Bres(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = Conv_kxnysz(i=3, k=9, n=64, s=1, if_bn=False, if_relu=(True, 1))
        self.Bres = self.makeLayers(nBres, 5)
        self.BresOutput = Conv_kxnysz(i=64, k=3, n=64, s=1, if_bn=True, if_relu=(False, 1))
        self.out1 =nn.Sequential(
            Conv_kxnysz(i=64, k=3, n=256, s=1, if_bn=False, if_relu=(True, 1), pixlS=True),
            Conv_kxnysz(i=64, k=3, n=256, s=1, if_bn=False, if_relu=(True, 1), pixlS=True)
        )

        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)

    def makeLayers(self, block, number_of_layers):
        layers = []
        for _ in range(number_of_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = out = self.input(x)
        out = self.Bres(out)
        out = self.BresOutput(out)
        out = torch.add(out, out1)
        out = self.out1(out)
        out = self.out(out)
        return out


class nLayers(nn.Module):
    def __init__(self):
        super(nLayers, self).__init__()
        self.Layers = nn.Sequential(
            Conv_kxnysz(i=64,  k=3, n=64,  s=2, if_bn=True,  if_relu=(True, 2),padding=False),
            Conv_kxnysz(i=64,  k=3, n=128, s=1, if_bn=True,  if_relu=(True, 2),padding=False),
            Conv_kxnysz(i=128, k=3, n=128, s=2, if_bn=True,  if_relu=(True, 2),padding=False),
            Conv_kxnysz(i=128, k=3, n=256, s=1, if_bn=True,  if_relu=(True, 2),padding=False),
            Conv_kxnysz(i=256, k=3, n=256, s=2, if_bn=True,  if_relu=(True, 2),padding=False),
            Conv_kxnysz(i=256, k=3, n=512, s=1, if_bn=True,  if_relu=(True, 2),padding=False),
            Conv_kxnysz(i=512, k=3, n=512, s=2, if_bn=True,  if_relu=(True, 2),padding=False)
        )

    def forward(self, x):
        return self.Layers(x)


class DisNet(nn.Module):
    def __init__(self, size):
        super(DisNet, self).__init__()
        self.input = Conv_kxnysz(i=3, k=3, n=64, s=1, if_bn=False, if_relu=(True, 2))
        self.layers = nLayers()
        self.dense1024 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
        self.LeakyReLU = nn.LeakyReLU()
        self.dense1 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, bias=True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.input(x)
        out = self.layers(out)
        out = self.dense1024(out)
        out = self.LeakyReLU(out)
        out = self.dense1(out)
        out = self.Sigmoid(out)
        return out

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x