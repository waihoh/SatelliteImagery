"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

nonlinearity = partial(F.relu, inplace=True)


class Dblock(nn.Module):
    """
    NOTE:
        Merged Dblock and Dblock_more_dilate.
        It's similar to Dblock class in dunet, but x is added in the forward function in this class.
        This looks like a residual block.
    """
    def __init__(self, channel, more_dilate=False):
        super().__init__()
        self.more_dilate = more_dilate
        self.dilate1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=8, padding=8)
        if more_dilate:
            self.dilate5 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, dilation=16, padding=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        if self.more_dilate:
            dilate5_out = nonlinearity(self.dilate5(dilate4_out))
            out = out + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(num_features=in_channels//4)

        self.deconv2 = nn.ConvTranspose2d(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels//4)

        self.conv3 = nn.Conv2d(in_channels=in_channels//4, out_channels=n_filters, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(num_features=n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = nonlinearity(self.norm1(x))
        x = self.deconv2(x)
        x = nonlinearity(self.norm2(x))
        x = self.conv3(x)
        x = nonlinearity(self.norm3(x))
        return x


class DLinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        filters = [64, 128, 256]
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock(channel=256, more_dilate=True)

        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(in_channels=filters[0], out_channels=32, kernel_size=4, stride=2, padding=1)
        self.finalconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.finalconv3 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final classification
        out = nonlinearity(self.finaldeconv1(d1))
        out = nonlinearity(self.finalconv2(out))
        out = torch.sigmoid(self.finalconv3(out))  # F.sigmoid is deprecated. Use torch.sigmoid instead.

        return out


class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4  # this layer is not present in DLinkNet34_less_pool

        self.dblock = Dblock(channel=512)  # this layer is with more_dilate=True in DLinkNet34_less_pool

        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=filters[2])  # this layer is not present in DLinkNet34_less_pool
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(in_channels=filters[0], out_channels=32, kernel_size=4, stride=2, padding=1)
        self.finalconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.finalconv3 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = nonlinearity(self.finaldeconv1(d1))
        out = nonlinearity(self.finalconv2(out))
        out = torch.sigmoid(self.finalconv3(out))  # F.sigmoid is deprecated. Use torch.sigmoid instead.

        return out


class DLinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(channel=2048, more_dilate=True)

        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(in_channels=filters[0], out_channels=32, kernel_size=4, stride=2, padding=1)
        self.finalconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.finalconv3 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = nonlinearity(self.finaldeconv1(d1))
        out = nonlinearity(self.finalconv2(out))
        out = torch.sigmoid(self.finalconv3(out))  # F.sigmoid is deprecated. Use torch.sigmoid instead.

        return out


class DLinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(channel=2048, more_dilate=True)

        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(in_channels=filters[0], out_channels=32, kernel_size=4, stride=2, padding=1)
        self.finalconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.finalconv3 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = nonlinearity(self.finaldeconv1(d1))
        out = nonlinearity(self.finalconv2(out))
        out = torch.sigmoid(self.finalconv3(out))  # F.sigmoid is deprecated. Use torch.sigmoid instead.

        return out


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # there is no center Dblock in LinkNet34

        self.decoder4 = DecoderBlock(in_channels=filters[3], n_filters=filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2], n_filters=filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1], n_filters=filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0], n_filters=filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(in_channels=filters[0], out_channels=32, kernel_size=4, stride=2, padding=1)
        self.finalconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.finalconv3 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = nonlinearity(self.finaldeconv1(d1))
        out = nonlinearity(self.finalconv2(out))
        out = torch.sigmoid(self.finalconv3(out))  # F.sigmoid is deprecated. Use torch.sigmoid instead.

        return out


if __name__ == "__main__":
    # mdl = DLinkNet34_less_pool()
    # mdl = DLinkNet34()
    # mdl = DLinkNet50()
    # mdl = DLinkNet101()
    mdl = LinkNet34()
    print(mdl)

    input = torch.rand(2, 3, 1024, 1024)
    output = mdl.forward(input)
    print(output)
    print(output.shape)
