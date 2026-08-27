import os
import glob
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import DataLoader, CacheDataset, ITKReader
from monai.networks.nets import SegResNet, resnet18  # POPRAWKA C: resnet18
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd, MapTransform,
    SpatialPadd, AsDiscrete, ScaleIntensityRangePercentilesd, EnsureTyped,
    RandRotate90d, RandScaleIntensityd, RandShiftIntensityd, RandAffined, Resized
)
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- 0. Konfiguracja bazowa ---
CROP_SIZE = 128
CLS_SIZE = 128          # Rozmiar dla globalnej klasyfikacji
NUM_CLASSES_CLS = 4     # 0: Contiguous-Dense, 1: Contiguous-Sparse, 2: Isolated-Dense, 3: Isolated-Sparse
NUM_CLASSES_SEG = 2     # 0: tło, 1: struktura
BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Uruchomiono na urządzeniu: {DEVICE}")

random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- 1. Customowe transformacje i Loss ---
class BinarizeLabeld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0).float()
        return d

class ClassificationFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(ClassificationFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=self.weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- 2. Ładowanie i podział danych ---
base_dir = "../data/SELMA3D2026_training_annotated"
categories = ["contiguous_structures", "isolated_structures"]

val_filenames_substrings = [
    "cell_nuclei_patchvolume_001", "cell_nuclei_patchvolume_002",
    "cfos_neuron_patchvolume_695", "cfos_neuron_patchvolume_1235",
    "ad_plaques_patchvolume_005", "ad_plaques_patchvolume_009",
    "ad_plaques_patchvolume_014", "ad_plaques_patchvolume_019",
    "blood_vessel_patchvolume_014", "blood_vessel_patchvolume_013",
    "blood_vessel_patchvolume_015", "human_nefh_transition",
    "macaque_pv_wm_1", "human_nefl_gm_2",
    "peripheral_nerve_patchvolume_289", "peripheral_nerve_patchvolume_395",
    "peripheral_nerve_patchvolume_1368"
]

train_files = []
val_files = []

for category in categories:
    raw_dir = os.path.join(base_dir, category, "raw")
    gt_dir = os.path.join(base_dir, category, "gt")
    
    raw_list = sorted(glob.glob(os.path.join(raw_dir, "*.mha")))
    gt_list = sorted(glob.glob(os.path.join(gt_dir, "*.mha")))
    
    for raw, gt in zip(raw_list, gt_list):
        filename = os.path.basename(raw).lower()
        density = "dense" if any(k in filename for k in ["cell_nuclei", "cfos_neuron", "blood_vessel"]) else "sparse"
        structure = "contiguous" if "contiguous" in category else "isolated"
        
        if structure == "contiguous" and density == "dense": class_id = 0
        elif structure == "contiguous" and density == "sparse": class_id = 1
        elif structure == "isolated" and density == "dense": class_id = 2
        else: class_id = 3
            
        data_dict = {"image": raw, "label": gt, "class_id": class_id}
        
        if any(v in filename for v in val_filenames_substrings):
            val_files.append(data_dict)
        else:
            train_files.append(data_dict)

random.shuffle(train_files)
print(f"Liczba plików treningowych: {len(train_files)} | Liczba plików walidacyjnych: {len(val_files)}")

train_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for item in train_files:
    train_class_counts[item["class_id"]] += 1

class_weights = [len(train_files) / (NUM_CLASSES_CLS * train_class_counts[i]) if train_class_counts[i] > 0 else 0.0 for i in range(NUM_CLASSES_CLS)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Liczebność klas (0-3): {train_class_counts}")

# --- 3. Transformacje (Rozdzielone dla CLS i SEG) ---
train_transforms_cls = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label", "class_id"]),
    BinarizeLabeld(keys=["label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Resized(keys=["image", "label"], spatial_size=(CLS_SIZE, CLS_SIZE, CLS_SIZE), mode=("trilinear", "nearest")),
    RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
])

val_transforms_cls = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label", "class_id"]),
    BinarizeLabeld(keys=["label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True, relative=False),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Resized(keys=["image", "label"], spatial_size=(CLS_SIZE, CLS_SIZE, CLS_SIZE), mode=("trilinear", "nearest")),
])

# Transformacje dla Segmentacji (Pozostały na patchach)
train_transforms_seg = Compose([
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
    RandAffined(keys=["image", "label"], prob=0.5, spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE), rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
])

val_transforms_seg = Compose([
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

# Dataloadery z rozbiciem na fazy
train_ds_cls = CacheDataset(data=train_files, transform=train_transforms_cls, cache_rate=1.0)
train_loader_cls = DataLoader(train_ds_cls, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_ds_cls = CacheDataset(data=val_files, transform=val_transforms_cls, cache_rate=1.0)
val_loader_cls = DataLoader(val_ds_cls, batch_size=1, num_workers=4)

train_ds_seg = CacheDataset(data=train_files, transform=train_transforms_seg, cache_rate=1.0)
train_loader_seg = DataLoader(train_ds_seg, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_ds_seg = CacheDataset(data=val_files, transform=val_transforms_seg, cache_rate=1.0)
val_loader_seg = DataLoader(val_ds_seg, batch_size=1, num_workers=4)

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

model_cls = resnet18(
    spatial_dims=3, 
    n_input_channels=1, 
    num_classes=NUM_CLASSES_CLS
).to(DEVICE)

model_cls_path = os.path.join(model_dir, "best_metric_model_cls_4.pth")
max_epochs_cls = 100

if os.path.exists(model_cls_path):
    model_cls.load_state_dict(torch.load(model_cls_path))
    log("Wczytano wagi klasyfikatora. Przechodzę do segmentacji.")
else:
    optimizer_cls = torch.optim.AdamW(model_cls.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler_cls = ReduceLROnPlateau(optimizer_cls, mode="max", patience=3, factor=0.5)
    loss_function_cls = ClassificationFocalLoss(weight=class_weights_tensor, gamma=2.0)
    scaler_cls = torch.amp.GradScaler(device='cuda')
    
    best_acc = -1.0
    patience_cls = 30
    epoch_no_improve_cls = 0
    
    for epoch in range(max_epochs_cls):
        model_cls.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader_cls:
            step += 1
            inputs, labels_cls = batch_data["image"].to(DEVICE), batch_data["class_id"].to(DEVICE)
            
            if labels_cls.dim() == 1 and labels_cls.shape[0] != inputs.shape[0]:
                labels_cls = labels_cls.expand(inputs.shape[0])
                
            optimizer_cls.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model_cls(inputs)
                loss = loss_function_cls(outputs, labels_cls)
            
            scaler_cls.scale(loss).backward()
            scaler_cls.step(optimizer_cls)
            scaler_cls.update()
            epoch_loss += loss.item()
            
        epoch_loss /= step
        
        if (epoch + 1) % 5 == 0:
            log(f"Faza 1 - Epoka {epoch + 1}/{max_epochs_cls} | Strata (Loss): {epoch_loss:.4f} | LR: {scheduler_cls.get_last_lr()[0]:.6f}")
            
            model_cls.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_data in val_loader_cls:
                    val_inputs = val_data["image"].to(DEVICE)
                    val_labels_cls = val_data["class_id"].to(DEVICE)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model_cls(val_inputs)
                    pred_class = torch.argmax(logits, dim=1)
                    
                    if pred_class.item() == val_labels_cls.item():
                        correct += 1
                    total += 1
                    
            acc = correct / total
            log(f"> Faza 1 Ewaluacja | Dokładność (Accuracy): {acc:.4f}")
            scheduler_cls.step(acc)
            
            if acc >= best_acc:
                best_acc = acc
                torch.save(model_cls.state_dict(), model_cls_path)
                log(f"> Zapisano nowy najlepszy klasyfikator (Acc: {best_acc:.4f})")
                epoch_no_improve_cls = 0
            else:
                epoch_no_improve_cls += 5
                if epoch_no_improve_cls >= patience_cls:
                    log(f"> Przerwano trenowanie klasyfikatora z powodu braku poprawy")
                    break

model_cls.load_state_dict(torch.load(model_cls_path))
model_cls.eval()
for param in model_cls.parameters():
    param.requires_grad = False

# =============================================================================
# FAZA 2: TRENING MODELU SEGMENTACYJNEGO
# =============================================================================
log(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START FAZY 2: SEGMENTACJA KASKADOWA")

model_seg = SegResNet(
    spatial_dims=3,
    in_channels=1 + NUM_CLASSES_CLS, 
    out_channels=NUM_CLASSES_SEG,
    init_filters=32,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    dropout_prob=0.2,
).to(DEVICE)

model_seg_path = os.path.join(model_dir, "best_metric_model_baseline.pth")
max_epochs_seg = 1000
val_interval = 10

if os.path.exists(model_seg_path):
    model_seg.load_state_dict(torch.load(model_seg_path))

optimizer_seg = torch.optim.AdamW(model_seg.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler_seg = ReduceLROnPlateau(optimizer_seg, mode="max", patience=5, factor=0.5)
loss_function_seg = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, gamma=2.0)
dsc_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES_SEG)
post_label = AsDiscrete(to_onehot=NUM_CLASSES_SEG)
scaler_seg = torch.amp.GradScaler(device='cuda')

best_metric = -1
best_metric_epoch = -1
patience = 200
epoch_no_improve = 0

for epoch in range(max_epochs_seg):
    model_seg.train()
    epoch_loss = 0
    step = 0
    
    # Używamy Loadera dedykowanego SEGMENTACJI (patche)
    for batch_data in train_loader_seg:
        step += 1
        inputs, labels_seg = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
        
        B, _, D, H, W = inputs.shape
        true_classes = batch_data["class_id"].to(DEVICE).long().view(-1)
        if true_classes.shape[0] == 1 and B > 1:
            true_classes = true_classes.expand(B)
            
        class_channels = torch.zeros((B, NUM_CLASSES_CLS, D, H, W), device=DEVICE, dtype=inputs.dtype)
        for b in range(B):
            class_channels[b, true_classes[b], ...] = 1.0
            
        seg_inputs = torch.cat([inputs, class_channels], dim=1)
        
        optimizer_seg.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model_seg(seg_inputs)
            loss = loss_function_seg(outputs, labels_seg)
        
        scaler_seg.scale(loss).backward()
        scaler_seg.step(optimizer_seg)
        scaler_seg.update()
        
        epoch_loss += loss.item()

    epoch_loss /= step
    if (epoch + 1) % 5 == 0:
        log(f"Faza 2 - Epoka {epoch + 1}/{max_epochs_seg} | Strata (Loss): {epoch_loss:.4f} | LR: {scheduler_seg.get_last_lr()[0]:.6f}")

    if (epoch + 1) % val_interval == 0:
        model_seg.eval()
        with torch.no_grad():
            for val_data in val_loader_seg: # Loader SEG podaje wielkie, oryginalne wolumeny
                val_inputs, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
                
                # POPRAWKA A (Spójność skali w inferencji kaskadowej):
                # Zamiast sliding window na dużym zdjęciu, skalujemy do rozmiaru na którym klasyfikator się uczył.
                val_inputs_cls_resized = F.interpolate(val_inputs, size=(CLS_SIZE, CLS_SIZE, CLS_SIZE), mode="trilinear", align_corners=False)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model_cls(val_inputs_cls_resized)
                pred_class = torch.argmax(logits, dim=1)
                
                # (Zgodnie z prośbą, zaimplementowano wyżej B w sliding_window_classification. 
                # Jeśli chcesz użyć starej logiki w zastępstwie interpolacji, użyj tej linii zamiast powyższych)
                # avg_logits = sliding_window_classification(val_inputs, model_cls, roi_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE))
                # pred_class = torch.argmax(avg_logits, dim=1)
                
                B_val, _, D_val, H_val, W_val = val_inputs.shape
                class_channels = torch.zeros((B_val, NUM_CLASSES_CLS, D_val, H_val, W_val), device=DEVICE, dtype=val_inputs.dtype)
                for b in range(B_val):
                    class_channels[b, pred_class[b], ...] = 1.0
                
                seg_inputs = torch.cat([val_inputs, class_channels], dim=1)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    val_outputs = sliding_window_inference(
                        seg_inputs, roi_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE), 
                        sw_batch_size=4, predictor=model_seg, overlap=0.5
                    )
                
                val_outputs = [post_pred(i) for i in val_outputs]
                val_labels = [post_label(i) for i in val_labels]
                dsc_metric(y_pred=val_outputs, y=val_labels)
                
            metric_tensor, not_nans = dsc_metric.aggregate()
            dsc_score = metric_tensor.item()
            dsc_metric.reset()
            
            log(f"> Faza 2 Ewaluacja Epoka {epoch + 1} | DSC: {dsc_score:.4f}")
            scheduler_seg.step(dsc_score)
            
            if dsc_score >= best_metric:
                best_metric = dsc_score
                best_metric_epoch = epoch + 1
                torch.save(model_seg.state_dict(), model_seg_path)
                log(f"> Zapisano nowy najlepszy segmentator (DSC: {best_metric:.4f})")
                epoch_no_improve = 0
            else:
                epoch_no_improve += val_interval
                if epoch_no_improve >= patience:
                    log(f"> Przerwano trenowanie z powodu braku poprawy przez {patience} epok")
                    break

log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ZAKOŃCZONO. Najlepsza metryka segmentacji (DSC): {best_metric:.4f} (Epoka {best_metric_epoch})")