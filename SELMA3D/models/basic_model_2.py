import os
import glob
import torch
from monai.data import DataLoader, CacheDataset
from monai.networks.layers import Norm
from monai.networks.nets import UNet
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
    RandAdjustContrastd,
    RandRotated, RandZoomd, Rand3DElasticd
)
from monai.data import ITKReader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import random

CROP_SIZE = 128

# Niestandardowa transformacja do mapowania binarnej maski na odpowiednią klasę
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

# 1. Konfiguracja ścieżek i przypisanie identyfikatorów klas
base_dir = "../SELMA3D2026_training_annotated"
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

random.seed(42)
random.shuffle(data_dicts)


val_percent = 0.20
val_size = int(len(data_dicts) * val_percent)

val_files = data_dicts[:val_size]
train_files = data_dicts[val_size:]

# 2. Definicja transformacji z dodanymi augmentacjami (szum i kontrast)
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader=ITKReader),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiClassd(keys=["label"], class_key="class_id"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),

        RandRotated(
            keys=["image", "label"],
            range_x=0.3,
            range_y=0.3,
            range_z=0.3,
            prob=0.3,
            mode=("bilinear", "nearest"),
        ),
        
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.8,
            max_zoom=1.2,
            prob=0.3,
            mode=("bilinear", "nearest"),
        ),
        
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            prob=0.2,
            mode=("bilinear", "nearest"),
        ),
        
        # Nowe augmentacje odpornościowe (szum i zmiany kontrastu)
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.1),
        RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.5, 2.0)),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader=ITKReader),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertToMultiClassd(keys=["label"], class_key="class_id"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        
        SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
        
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
# 3. Zbiory danych i DataLoadery
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# 4. Definicja modelu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.INSTANCE,
    dropout=0.2
).to(device)
'''
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.INSTANCE,
    dropout=0.2
).to(device)

model_path = "../trained_models/best_metric_model_unet_bigger.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Wczytano wagi z pliku: {model_path} – kontynuacja treningu.")
else:
    print("Brak pliku wag. Rozpoczęcie treningu od zera.")
print(f"Model załadowany. Urządzenie: {device}")

# 5. Konfiguracja parametrów treningu, optymalizatora i schedulera
max_epochs = 1000
val_interval = 25
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# Scheduler: redukuje współczynnik uczenia o połowę, jeśli metryka nie poprawi się przez 10 walidacji
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)

print("Rozpoczęcie treningu na urządzeniu:", device)

# 6. Pętla ucząca
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"Epoka {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        #print(f"{step}/{len(train_loader)}, strata treningowa: {loss.item():.4f}")
        
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Średnia strata z epoki: {epoch_loss:.4f}")

    # 7. Walidacja
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                roi_size = (CROP_SIZE, CROP_SIZE, CROP_SIZE)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                
                val_outputs = [post_pred(i) for i in val_outputs]
                val_labels = [post_label(i) for i in val_labels]
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            # Aktualizacja schedulera na podstawie metryki walidacyjnej
            scheduler.step(metric)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "../trained_models/best_metric_model_unet_bigger.pth")
                print("Zapisano nowy najlepszy model.")
                
            print(
                f"Obecna metryka (Mean Dice): {metric:.4f} "
                f"Najlepsza metryka: {best_metric:.4f} w epoce: {best_metric_epoch}"
            )

print(f"Zakończono trening. Najlepsza metryka: {best_metric:.4f} w epoce {best_metric_epoch}")