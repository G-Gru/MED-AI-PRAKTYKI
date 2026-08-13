from monai.networks.nets import SegResNet, UNet, BasicUNet
from monai.networks.layers import Norm

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