import os
import glob
import random
from monai.data import DataLoader, CacheDataset, ITKReader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd, MapTransform,
    SpatialPadd, ScaleIntensityRangePercentilesd, RandSpatialCropd,
    RandCoarseDropoutd, CopyItemsd
)
class ConvertToMultiClassd(MapTransform):
    def __init__(self, keys, class_key="class_id", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.class_key = class_key

    def __call__(self, data):
        d = dict(data)
        class_id = d[self.class_key]
        for key in self.keys:
            d[key] = d[key] * class_id
        return d


# SCANNERS
def get_supervised_file_dicts(base_dir, val_percent=0.20, seed=42):
    categories = {
        "contiguous_structures": 1,
        "isolated_structures": 2
    }
    data_dicts = []
    for category, class_id in categories.items():
        raw_dir = os.path.join(base_dir, category, "raw")
        gt_dir = os.path.join(base_dir, category, "gt")

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.mha")))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.mha")))

        for raw, gt in zip(raw_files, gt_files):
            data_dicts.append({"image": raw, "label": gt, "class_id": class_id})

    random.seed(seed)
    random.shuffle(data_dicts)

    val_size = int(len(data_dicts) * val_percent)
    return data_dicts[val_size:], data_dicts[:val_size]


def get_ssl_file_dicts(base_dir, val_percent=0.20, seed=42):
    nii_files = sorted(glob.glob(os.path.join(base_dir, "**", "*.nii.gz"), recursive=True))
    nii_files = [f for f in nii_files if os.path.isfile(f)]
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in base_dir: {base_dir}")

    data_dicts = [{"image": f} for f in nii_files]

    random.seed(seed)
    random.shuffle(data_dicts)

    val_size = int(len(data_dicts) * val_percent)
    return data_dicts[val_size:], data_dicts[:val_size]


# TRANSFORMS
def get_supervised_transforms(crop_size=128, is_train=True, resize=False):
    target_pixdim = (2.0, 2.0, 2.0) if resize else (1.0, 1.0, 1.0)
    transforms = [
        LoadImaged(keys=["image", "label"], reader=ITKReader),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiClassd(keys=["label"], class_key="class_id"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(crop_size, crop_size, crop_size)),
    ]

    if is_train:
        transforms.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(crop_size, crop_size, crop_size),
                pos=1, neg=1, num_samples=4,
                image_key="image", image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        ])

    return Compose(transforms)


def get_ssl_transforms(crop_size=128, hole_size=32, is_train=True, resize = False):
    target_pixdim = (2.0, 2.0, 2.0) if resize else (1.0, 1.0, 1.0)
    transforms = [
        LoadImaged(keys=["image"], reader=ITKReader),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=target_pixdim, mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(crop_size, crop_size, crop_size)),
    ]

    if is_train:
        transforms.extend([
            RandSpatialCropd(keys=["image"], roi_size=(crop_size, crop_size, crop_size), random_size=False),
            RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
        ])

    transforms.append(CopyItemsd(keys=["image"], times=1, names=["label"]))
    #transforms.append(RandCoarseDropoutd(keys=["image"], holes=4, spatial_size=hole_size, fill_value=0.0, prob=1.0))

    return Compose(transforms)


# DATALOADER
def get_dataloader(data_list, transforms, batch_size, num_workers=4, shuffle=True, to_ram=True):
    if to_ram:
        ds = CacheDataset(data=data_list, transform=transforms, cache_rate=1.0)
    else:
        ds = Dataset(data=data_list, transform=transforms)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )