import os
import glob
import torch
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    MapTransform,
    SpatialPadd,
    RandGaussianNoised,
    RandAdjustContrastd
)
from monai.data import ITKReader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import random

print(f"\nPyTorch Version: {torch.__version__}\n")

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Built with CUDA: {torch.version.cuda}")
print(f"Device Count: {torch.cuda.device_count()}\n")

print(f"MPS Available:{torch.mps.is_available()}")
print(f"Device Count: {torch.mps.device_count()}\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using: ",device)
