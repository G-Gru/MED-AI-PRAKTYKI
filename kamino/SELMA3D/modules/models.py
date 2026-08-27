from monai.networks.nets import SegResNet, UNet, BasicUNet, AutoEncoder
from monai.networks.layers import Norm
import torch
import torch.nn as nn
import torch.nn.functional as F


# CUSTOM MODELS

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DynamicUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None, phase2=False):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.skip_connections_allowed = phase2
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        prev_channels = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(prev_channels, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
            prev_channels = feature

        if phase2:
            self.segmentation_head = nn.Conv3d(features[0], out_channels, kernel_size=1)

            for param in self.parameters():
                param.requires_grad = False

            for param in self.segmentation_head.parameters():
                param.requires_grad = True
        else:
            self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

            if self.skip_connections_allowed:
                x = torch.cat((skip, x), dim=1)
            else:
                x = torch.cat((torch.zeros_like(skip), x), dim=1)

            x = self.ups[i + 1](x)

        if self.skip_connections_allowed:
            return self.segmentation_head(x)
        else:
            return self.final_conv(x)


# FUNCTIONS TO GET MODELS


def get_segresnet(init_filters=16):
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        init_filters=init_filters
    )

def get_segresnet_ssl(init_filters=16):
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        init_filters=init_filters
    )

def get_unet(channels = (16, 32, 64, 128, 256), strides = (2, 2, 2, 2), num_res_units = 2, norm = Norm.INSTANCE, dropout = 0.2):
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels= channels,
        strides= strides,
        num_res_units= num_res_units,
        norm= norm,
        dropout= dropout,
    )

def get_unet_ssl(channels = (16, 32, 64, 128, 256), strides = (2, 2, 2, 2), num_res_units = 2, norm = Norm.INSTANCE, dropout = 0.2):
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels= channels,
        strides= strides,
        num_res_units= num_res_units,
        norm= norm,
        dropout= dropout,
    )

def get_basicunet(features):
    return BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        features=(32, 32, 64, 128, 256, 32)
    )

def get_basicunet_ssl(features):
    return BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32)
    )

def get_autoencoder_ssl(features=(16, 32, 64, 128)):
    return AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=features,
        strides=(1, 2, 2, 2),
    )

def get_custom_unet(features=(16, 32, 64, 128, 256), phase2 = False, out_channels = 1):
    return DynamicUNet3D(
        in_channels=1,
        out_channels=out_channels,
        features=features,
        phase2=phase2
    )