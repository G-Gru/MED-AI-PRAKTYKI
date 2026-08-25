import os
import glob
import random
import datetime
import torch

from monai.data import DataLoader, CacheDataset, ITKReader
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd, MapTransform,
    SpatialPadd, AsDiscrete, ScaleIntensityRangePercentilesd
)
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingLR

# Konfiguracja bazowa
CROP_SIZE = 128
NUM_CLASSES = 3  # 0: tło, 1: contiguous_structures, 2: isolated_structures
BATCH_SIZE = 2

class ConvertToMultiClassd(MapTransform):
    def __init__(self, keys, class_key="class_id", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.class_key = class_key

    def __call__(self, data):
        d = dict(data)
        class_id = d[self.class_key]
        for key in self.keys:
            d[key] = (d[key] > 0).float() * class_id
        return d

# 1. Ładowanie i podział danych
base_dir = "./SELMA3D2026_training_annotated"
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

# 2. Transformacje
train_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiClassd(keys=["label"], class_key="class_id"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE), pos=1, neg=1, num_samples=4, image_key="image"),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiClassd(keys=["label"], class_key="class_id"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
])

# 3. DataLoadery
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# 4. Model i środowisko
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=NUM_CLASSES,
    init_filters=16,
    dropout_prob=0.2,
).to(device)

model_dir = "./trained_models"
model_name = "best_metric_model_segresnet"
model_path = os.path.join(model_dir, f"{model_name}.pth")
log_path = os.path.join(model_dir, f"log-{model_name}.txt")

os.makedirs(model_dir, exist_ok=True)

def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START TRENINGU")
log(f"Model: SegResNet (init_filters=16, dropout_prob=0.2)")
log(f"Loss Function: DiceFocalLoss (gamma=2.0, to_onehot_y=True, softmax=True, include_background=False)")
log(f"Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)")
log(f"Scheduler: CosineAnnealingLR (T_max=1000, eta_min=1e-6)")
log(f"Batch Size: {BATCH_SIZE}")
log(f"Crop Size: {CROP_SIZE}, Validation Overlap: 0.5")
log("-" * 50)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    log(f"Wczytano wagi z pliku: {model_path} – kontynuacja treningu.")
else:
    log("Brak pliku wag. Rozpoczęcie treningu od zera.")

log(f"Urządzenie: {device}")

# 5. Konfiguracja hiperparametrów
max_epochs = 1000
val_interval = 10
best_metric = -1
best_metric_epoch = -1

loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
post_label = AsDiscrete(to_onehot=NUM_CLASSES)
scaler = torch.amp.GradScaler('cuda')

# 6. Pętla ucząca
for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

    epoch_loss /= step
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        log(f"Epoka {epoch + 1}/{max_epochs} | Strata (Loss): {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 7. Walidacja
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                with torch.amp.autocast('cuda'):
                    val_outputs = sliding_window_inference(
                        val_inputs, 
                        roi_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE), 
                        sw_batch_size=4, 
                        predictor=model,
                        overlap=0.5
                    )
                
                val_outputs = [post_pred(i) for i in val_outputs]
                val_labels = [post_label(i) for i in val_labels]
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            metric_tensor, not_nans = dice_metric.aggregate()
            metric = metric_tensor.item()
            dice_metric.reset()
            
            log(f"> Ewaluacja Epoka {epoch + 1} | Mean Dice: {metric:.4f}")
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)
                log(f"> Zapisano nowy najlepszy model (Dice: {best_metric:.4f})")

log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ZAKOŃCZONO. Najlepsza metryka: {best_metric:.4f} (Epoka {best_metric_epoch})")