import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

nonlinearity = partial(F.relu, inplace=True)


class Dblock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.dilate1 = nn.Conv2d(in_channels=int(channel/2), out_channels=channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=16, padding=16)
        self.dilate6 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=32, padding=32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        dilate6_out = nonlinearity(self.dilate6(dilate5_out))
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out + dilate6_out
        return out


class Dunet(nn.Module):
    def __init__(self):
        super().__init__()

        vgg13 = models.vgg13(weights='VGG13_Weights.DEFAULT')  # Changed to weights. pretrained parameter is deprecated

        self.conv1 = vgg13.features[0]
        self.conv2 = vgg13.features[2]
        self.conv3 = vgg13.features[5]
        self.conv4 = vgg13.features[7]
        self.conv5 = vgg13.features[10]
        self.conv6 = vgg13.features[12]

        self.dilate_center = Dblock(channel=512)

        self.up3 = self.conv_stage(dim_in=512, dim_out=256)
        self.up2 = self.conv_stage(dim_in=256, dim_out=128)
        self.up1 = self.conv_stage(dim_in=128, dim_out=64)

        self.trans3 = self.upsample(ch_coarse=512, ch_fine=256)
        self.trans2 = self.upsample(ch_coarse=256, ch_fine=128)
        self.trans1 = self.upsample(ch_coarse=128, ch_fine=64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=dim_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=dim_out),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(inplace=True),
            )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch_coarse, out_channels=ch_fine, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        stage1 = nonlinearity(self.conv2(nonlinearity(self.conv1(x))))
        stage2 = nonlinearity(self.conv4(nonlinearity(self.conv3(self.max_pool(stage1)))))
        stage3 = nonlinearity(self.conv6(nonlinearity(self.conv5(self.max_pool(stage2)))))

        out = self.dilate_center(self.max_pool(stage3))

        out = self.up3(torch.cat((self.trans3(out), stage3), 1))
        out = self.up2(torch.cat((self.trans2(out), stage2), 1))
        out = self.up1(torch.cat((self.trans1(out), stage1), 1))

        out = self.conv_last(out)

        return out


if __name__ == "__main__":
    mdl = Dunet()
    print(mdl)
