import os
import glob
import random
import datetime
import torch
import torch.nn as nn

from monai.data import DataLoader, CacheDataset, ITKReader
from monai.networks.nets import SegResNet, resnet10
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd, MapTransform,
    SpatialPadd, AsDiscrete, ScaleIntensityRangePercentilesd, EnsureTyped,
    RandRotate90d, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd
)
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- 0. Konfiguracja bazowa ---
CROP_SIZE = 128
NUM_CLASSES_CLS = 2  # 0: contiguous_structures, 1: isolated_structures
NUM_CLASSES_SEG = 2  # 0: tło, 1: struktura
BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Uruchomiono na urządzeniu: {DEVICE}")

# --- 1. Customowe transformacje ---
class BinarizeLabeld(MapTransform):
    """Sprowadza dowolne wartości > 0 do klasy 1 (foreground) oraz zostawia 0 (background)."""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0).float()
        return d

# --- 2. Ładowanie i podział danych ---
base_dir = "./SELMA3D2026_training_annotated"
categories = {
    "contiguous_structures": 0,
    "isolated_structures": 1
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

# --- 3. Transformacje i DataLoadery ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label", "class_id"]),
    BinarizeLabeld(keys=["label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE), pos=1, neg=1, num_samples=4, image_key="image"),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),

    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label", "class_id"]),
    BinarizeLabeld(keys=["label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# --- 4. System logowania ---
model_dir = "./trained_models"
os.makedirs(model_dir, exist_ok=True)
log_path = os.path.join(model_dir, "training_pipeline_log.txt")

def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# =============================================================================
# FAZA 1: TRENING MODELU KLASYFIKACYJNEGO
# =============================================================================
log(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START FAZY 1: KLASYFIKACJA")

model_cls = resnet10(
    spatial_dims=3, 
    n_input_channels=1, 
    num_classes=NUM_CLASSES_CLS
).to(DEVICE)

model_cls_path = os.path.join(model_dir, "best_metric_model_cls.pth")
max_epochs_cls = 100

if os.path.exists(model_cls_path):
    model_cls.load_state_dict(torch.load(model_cls_path))
    log("Wczytano wagi klasyfikatora. Przechodzę do segmentacji (możesz wymusić trening usuwając plik).")
else:
    optimizer_cls = torch.optim.AdamW(model_cls.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_function_cls = nn.CrossEntropyLoss()
    scaler_cls = torch.amp.GradScaler('cuda')
    
    best_acc = -1.0
    
    def sliding_window_classification(image_tensor, model, roi_size, overlap=0.5):
        """Funkcja realizująca sliding window dla klasyfikatora."""
        _, _, D, H, W = image_tensor.shape
        step_D = max(1, int(roi_size[0] * (1 - overlap)))
        step_H = max(1, int(roi_size[1] * (1 - overlap)))
        step_W = max(1, int(roi_size[2] * (1 - overlap)))

        logits_list = []
        for d in range(0, max(1, D - roi_size[0] + 1), step_D):
            for h in range(0, max(1, H - roi_size[1] + 1), step_H):
                for w in range(0, max(1, W - roi_size[2] + 1), step_W):
                    d_start, h_start, w_start = d, h, w
                    d_end = min(d_start + roi_size[0], D)
                    h_end = min(h_start + roi_size[1], H)
                    w_end = min(w_start + roi_size[2], W)
                    
                    # Jeśli okno wychodzi poza obraz, przesuwamy początek
                    d_start = max(0, d_end - roi_size[0])
                    h_start = max(0, h_end - roi_size[1])
                    w_start = max(0, w_end - roi_size[2])

                    patch = image_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        logits_list.append(model(patch))
        
        return torch.mean(torch.stack(logits_list), dim=0)

    for epoch in range(max_epochs_cls):
        model_cls.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels_cls = batch_data["image"].to(DEVICE), batch_data["class_id"].to(DEVICE)
            
            optimizer_cls.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model_cls(inputs)
                loss = loss_function_cls(outputs, labels_cls)
            
            scaler_cls.scale(loss).backward()
            scaler_cls.step(optimizer_cls)
            scaler_cls.update()
            
            epoch_loss += loss.item()
            
        epoch_loss /= step
        
        if (epoch + 1) % 5 == 0:
            log(f"Faza 1 - Epoka {epoch + 1}/{max_epochs_cls} | Strata (Loss): {epoch_loss:.4f}")
            
            # Ewaluacja Sliding Window
            model_cls.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(DEVICE)
                    val_labels_cls = val_data["class_id"].to(DEVICE)
                    
                    avg_logits = sliding_window_classification(val_inputs, model_cls, roi_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE))
                    pred_class = torch.argmax(avg_logits, dim=1)
                    
                    if pred_class.item() == val_labels_cls.item():
                        correct += 1
                    total += 1
                    
            acc = correct / total
            log(f"> Faza 1 Ewaluacja | Dokładność (Accuracy): {acc:.4f}")
            
            if acc >= best_acc:
                best_acc = acc
                torch.save(model_cls.state_dict(), model_cls_path)
                log(f"> Zapisano nowy najlepszy klasyfikator (Acc: {best_acc:.4f})")

# Przed przejściem do fazy 2 zamrażamy klasyfikator i ładujemy najlepsze wagi
model_cls.load_state_dict(torch.load(model_cls_path))
model_cls.eval()
for param in model_cls.parameters():
    param.requires_grad = False

# =============================================================================
# FAZA 2: TRENING MODELU SEGMENTACYJNEGO
# =============================================================================
log(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START FAZY 2: SEGMENTACJA KASKADOWA")

# Zauważ: in_channels = 1 (obraz) + 2 (one-hot mapy klas) = 3
model_seg = SegResNet(
    spatial_dims=3,
    in_channels=1 + NUM_CLASSES_CLS,
    out_channels=NUM_CLASSES_SEG,
    init_filters=32,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    dropout_prob=0.2,
).to(DEVICE)

model_seg_path = os.path.join(model_dir, "best_metric_model_segresnet_cascade.pth")
max_epochs_seg = 1000
val_interval = 10

if os.path.exists(model_seg_path):
    model_seg.load_state_dict(torch.load(model_seg_path))
    log("Wczytano wagi segmentatora kaskadowego. Kontynuacja treningu.")

optimizer_seg = torch.optim.AdamW(model_seg.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler_seg = CosineAnnealingWarmRestarts(optimizer_seg, T_0=50, T_mult=2, eta_min=1e-6)
loss_function_seg = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, gamma=2.0)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES_SEG)
post_label = AsDiscrete(to_onehot=NUM_CLASSES_SEG)
scaler_seg = torch.amp.GradScaler('cuda')

# Klasa Predictora łącząca oba modele pod sliding_window_inference
class CascadePredictor(nn.Module):
    def __init__(self, classifier, segmenter, num_classes):
        super().__init__()
        self.classifier = classifier
        self.segmenter = segmenter
        self.num_classes = num_classes

    def forward(self, x):
        with torch.no_grad():
            cls_logits = self.classifier(x)
            pred_classes = torch.argmax(cls_logits, dim=1)
            
            B, _, D, H, W = x.shape
            class_channels = torch.zeros((B, self.num_classes, D, H, W), device=x.device, dtype=x.dtype)
            for b in range(B):
                class_channels[b, pred_classes[b], ...] = 1.0
                
        x_concat = torch.cat([x, class_channels], dim=1)
        return self.segmenter(x_concat)

best_metric = -1
best_metric_epoch = -1
patience = 100
epoch_no_improve = 0

for epoch in range(max_epochs_seg):
    model_seg.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels_seg = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
        
        # 1. Przewidujemy klasę obrazu zamrożonym klasyfikatorem
        with torch.no_grad(), torch.amp.autocast('cuda'):
            cls_logits = model_cls(inputs)
            pred_classes = torch.argmax(cls_logits, dim=1)
            
        # 2. Tworzymy kanały przestrzenne dla przewidzianej klasy
        B, _, D, H, W = inputs.shape
        class_channels = torch.zeros((B, NUM_CLASSES_CLS, D, H, W), device=DEVICE, dtype=inputs.dtype)
        for b in range(B):
            class_channels[b, pred_classes[b], ...] = 1.0
            
        # 3. Łączymy obraz i mapę klas w jeden tensor 
        seg_inputs = torch.cat([inputs, class_channels], dim=1)
        
        # 4. Trening segmentatora
        optimizer_seg.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model_seg(seg_inputs)
            loss = loss_function_seg(outputs, labels_seg)
        
        scaler_seg.scale(loss).backward()
        scaler_seg.step(optimizer_seg)
        scaler_seg.update()
        
        epoch_loss += loss.item()

    epoch_loss /= step
    scheduler_seg.step()
    
    if (epoch + 1) % 5 == 0:
        log(f"Faza 2 - Epoka {epoch + 1}/{max_epochs_seg} | Strata (Loss): {epoch_loss:.4f} | LR: {scheduler_seg.get_last_lr()[0]:.6f}")

    # Ewaluacja Segmentatora (Sliding Window Kaskadowe)
    if (epoch + 1) % val_interval == 0:
        cascade_predictor = CascadePredictor(model_cls, model_seg, NUM_CLASSES_CLS).eval()
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    val_outputs = sliding_window_inference(
                        val_inputs, 
                        roi_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE), 
                        sw_batch_size=4, 
                        predictor=cascade_predictor,
                        overlap=0.5
                    )
                
                val_outputs = [post_pred(i) for i in val_outputs]
                val_labels = [post_label(i) for i in val_labels]
                dice_metric(y_pred=val_outputs, y=val_labels)
                
            metric_tensor, not_nans = dice_metric.aggregate()
            metric = metric_tensor.item()
            dice_metric.reset()
            
            log(f"> Faza 2 Ewaluacja Epoka {epoch + 1} | Mean Dice: {metric:.4f}")
            
            if metric >= best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model_seg.state_dict(), model_seg_path)
                log(f"> Zapisano nowy najlepszy segmentator (Dice: {best_metric:.4f})")
                epoch_no_improve = 0
            else:
                epoch_no_improve += val_interval

                if epoch_no_improve >= patience:
                    log(f"> Przerwano trenowanie z powodu braku poprawy przez {patience} epok")
                    break

log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ZAKOŃCZONO. Najlepsza metryka segmentacji: {best_metric:.4f} (Epoka {best_metric_epoch})")