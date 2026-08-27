import os
import glob
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import DataLoader, CacheDataset, ITKReader
from monai.networks.nets import SegResNet, resnet18
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, NormalizeIntensityd, MapTransform,
    SpatialPadd, AsDiscrete, ScaleIntensityRangePercentilesd, EnsureTyped,
    RandRotate90d, RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    RandAffined, Resized
)
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =============================================================================
# [SEKCJA 1] KONFIGURACJA BAZOWA
# =============================================================================
CONFIG = {
    "data_dir": "../data/SELMA3D2026_training_annotated",
    "model_save_dir": "./trained_models",
    
    "crop_size": 128,        # Rozmiar patcha
    "global_size": 128,      # Rozmiar kompresji dla Struktury
    
    "batch_size_struct": 2,  # Batch dla modelu Struktury (skompresowany obraz = mało pamięci)
    "batch_size_dense": 1,   # Zmniejszony batch 
    "num_samples_dense": 2,  # Tylko 2 patche na obraz zamiast 4
    "batch_size_seg": 1,     # Batch dla segmentatora
    
    "epochs_cls": 300,
    "epochs_seg": 1000,
    
    "num_classes_struct": 2, 
    "num_classes_dense": 2,  
    "num_classes_combined": 4, 
    "num_classes_seg": 2,    
    
    "seed": 42
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Uruchomiono na urządzeniu: {DEVICE}")

random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
log_path = os.path.join(CONFIG["model_save_dir"], "training_pipeline_log.txt")

def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# =============================================================================
# [SEKCJA 2] CUSTOMOWE FUNKCJE STRATY I METRYKI
# =============================================================================
class BinarizeLabeld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0).float()
        return d

class ClassificationFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# =============================================================================
# [SEKCJA 3] LOGIKA ŁADOWANIA DANYCH I ETYKIETOWANIA
# =============================================================================
def build_dataset_dicts():
    categories = ["contiguous_structures", "isolated_structures"]
    val_filenames = [
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

    train_data, val_data = [], []
    for category in categories:
        raw_list = sorted(glob.glob(os.path.join(CONFIG["data_dir"], category, "raw", "*.mha")))
        gt_list = sorted(glob.glob(os.path.join(CONFIG["data_dir"], category, "gt", "*.mha")))
        
        for raw, gt in zip(raw_list, gt_list):
            fname = os.path.basename(raw).lower()
            
            is_dense = any(k in fname for k in ["cell_nuclei", "cfos_neuron", "blood_vessel"])
            is_contiguous = "contiguous" in category
            
            struct_id = 0 if is_contiguous else 1
            dense_id = 0 if is_dense else 1
            combined_id = (struct_id * 2) + dense_id 
            
            data_dict = {
                "image": raw, 
                "label": gt, 
                "struct_id": struct_id, 
                "dense_id": dense_id, 
                "combined_id": combined_id
            }
            
            if any(v in fname for v in val_filenames):
                val_data.append(data_dict)
            else:
                train_data.append(data_dict)

    random.shuffle(train_data)
    return train_data, val_data

def get_class_weights(train_data, key, num_classes):
    counts = {i: 0 for i in range(num_classes)}
    for item in train_data:
        counts[item[key]] += 1
        
    weights = [len(train_data) / (num_classes * counts[i]) if counts[i] > 0 else 0.0 for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE), counts

# =============================================================================
# [SEKCJA 4] FABRYKA TRANSFORMACJI (MODULARNA)
# =============================================================================
def get_transforms(mode="global", is_train=True):
    base_transforms = [
        LoadImaged(keys=["image", "label"], reader=ITKReader),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label", "struct_id", "dense_id", "combined_id"]),
        BinarizeLabeld(keys=["label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image")
    ]
    
    spatial_transforms = []
    if mode == "global":
        spatial_transforms.append(
            Resized(keys=["image", "label"], spatial_size=(CONFIG["global_size"],)*3, mode=("trilinear", "nearest"))
        )
    elif mode == "local":
        spatial_transforms.append(
            SpatialPadd(keys=["image", "label"], spatial_size=(CONFIG["crop_size"],)*3)
        )
        if is_train:
            spatial_transforms.append(
                RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", 
                                       spatial_size=(CONFIG["crop_size"],)*3, pos=1, neg=0, 
                                       num_samples=CONFIG["num_samples_dense"], image_key="image") # [POPRAWKA OOM] Mniej patchy na batch
            )
            
    aug_transforms = []
    if is_train:
        aug_transforms = [
            RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
            RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1)
        ]
        
    return Compose(base_transforms + spatial_transforms + aug_transforms)

# =============================================================================
# [SEKCJA 5] REUSABLE TRAINING LOOPS (LOGIKA TRENOWANIA)
# =============================================================================
def eval_density_sliding_window(image_tensor, label_tensor, model, roi_size):
    """
    Sliding window oceniający gęstość WYŁĄCZNIE na łatach zawierających tkankę.
    Wykorzystuje maskę (label_tensor) do bezwzględnego odrzucenia pustego tła.
    """
    _, _, D, H, W = image_tensor.shape
    sD, sH, sW = roi_size, roi_size, roi_size
    logits_list = []
    
    for d in range(0, max(1, D - sD + 1), sD // 2):
        for h in range(0, max(1, H - sH + 1), sH // 2):
            for w in range(0, max(1, W - sW + 1), sW // 2):
                d_e, h_e, w_e = min(d+sD, D), min(h+sH, H), min(w+sW, W)
                d_s, h_s, w_s = max(0, d_e-sD), max(0, h_e-sH), max(0, w_e-sW)
                
                patch_img = image_tensor[:, :, d_s:d_e, h_s:h_e, w_s:w_e]
                patch_lbl = label_tensor[:, :, d_s:d_e, h_s:h_e, w_s:w_e]
                
                if patch_lbl.sum() > 0:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(patch_img)
                        probs = torch.softmax(logits, dim=1)
                        logits_list.append(probs)
                        
    if not logits_list:
        mid_d, mid_h, mid_w = D//2, H//2, W//2
        patch_img = image_tensor[:, :, max(0, mid_d-sD//2):mid_d+sD//2, max(0, mid_h-sH//2):mid_h+sH//2, max(0, mid_w-sW//2):mid_w+sW//2]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(patch_img)
            probs = torch.softmax(logits, dim=1)
            logits_list.append(probs)

    return torch.mean(torch.stack(logits_list), dim=0)

def train_classifier(model, train_loader, val_loader, target_key, model_name, class_weights):
    save_path = os.path.join(CONFIG["model_save_dir"], f"best_metric_{model_name}.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        log(f"[ZNAZLEZIONO] Wczytano wagi dla: {model_name}. Pomijam trening.")
        model.eval()
        return model

    log(f"\n--- ROZPOCZĘCIE TRENINGU: {model_name.upper()} ---")
    if model_name == "density": 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    else: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.5)
    loss_fn = ClassificationFocalLoss(weight=class_weights, gamma=2.0)
    scaler = torch.amp.GradScaler(device='cuda')
    
    best_acc, epoch_no_improve, patience = -1.0, 0, 100
    
    for epoch in range(CONFIG["epochs_cls"]):
        model.train()
        epoch_loss, step = 0, 0
        
        for batch in train_loader:
            step += 1
            inputs, labels = batch["image"].to(DEVICE), batch[target_key].to(DEVICE)
            
            if labels.dim() == 1 and labels.shape[0] != inputs.shape[0]:
                labels = labels.expand(inputs.shape[0])
                
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / step
            log(f"[{model_name}] Epoka {epoch + 1} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_in, val_lbl = val_batch["image"].to(DEVICE), val_batch[target_key].to(DEVICE)
                    val_mask = val_batch["label"].to(DEVICE) 
                    
                    if "density" in model_name:
                        logits = eval_density_sliding_window(val_in, val_mask, model, CONFIG["crop_size"])
                    else:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(val_in) 
                            
                    pred = torch.argmax(logits, dim=1)
                    if pred.item() == val_lbl.item():
                        correct += 1
                    total += 1
                    
            acc = correct / total
            log(f" > Ewaluacja [{model_name}]: Accuracy = {acc:.4f}")
            scheduler.step(acc)
            
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
                log(f" > Zapisano najlepszy model {model_name} (Acc: {best_acc:.4f})")
            if acc <= best_acc:
                epoch_no_improve += 5
                if epoch_no_improve >= patience:
                    log(f" > Early stopping dla {model_name}.")
                    break
            else:
                epoch_no_improve = 0

    model.load_state_dict(torch.load(save_path))
    model.eval()
    return model

# =============================================================================
# [SEKCJA 6] GŁÓWNY POTOK WYKONAWCZY (PIPELINE)
# =============================================================================
if __name__ == "__main__":
    log(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START SYSTEMU")
    torch.cuda.empty_cache()

    # 1. PRZYGOTOWANIE DANYCH
    train_files, val_files = build_dataset_dicts()
    
    # 2. TRENOWANIE MODELU A (STRUKTURA)
    ds_train_struct = CacheDataset(train_files, transform=get_transforms("global", True), cache_rate=1.0)
    ds_val_struct   = CacheDataset(val_files, transform=get_transforms("global", False), cache_rate=1.0)
    dl_train_struct = DataLoader(ds_train_struct, batch_size=CONFIG["batch_size_struct"], shuffle=True)
    dl_val_struct   = DataLoader(ds_val_struct, batch_size=1)
    weights_struct, _ = get_class_weights(train_files, "struct_id", CONFIG["num_classes_struct"])

    model_struct = resnet18(spatial_dims=3, n_input_channels=1, num_classes=CONFIG["num_classes_struct"]).to(DEVICE)
    model_struct = train_classifier(model_struct, dl_train_struct, dl_val_struct, "struct_id", "model_structure", weights_struct)
    
    # Czyszczenie pamięci przed drugim modelem
    torch.cuda.empty_cache()

    # 3. TRENOWANIE MODELU B (GĘSTOŚĆ)
    ds_train_dense = CacheDataset(train_files, transform=get_transforms("local", True), cache_rate=1.0)
    ds_val_dense   = CacheDataset(val_files, transform=get_transforms("local", False), cache_rate=1.0)
    dl_train_dense = DataLoader(ds_train_dense, batch_size=CONFIG["batch_size_dense"], shuffle=True)
    dl_val_dense   = DataLoader(ds_val_dense, batch_size=1)
    weights_dense, _ = get_class_weights(train_files, "dense_id", CONFIG["num_classes_dense"])

    model_dense = resnet18(spatial_dims=3, n_input_channels=1, num_classes=CONFIG["num_classes_dense"]).to(DEVICE)
    model_dense = train_classifier(model_dense, dl_train_dense, dl_val_dense, "dense_id", "model_density", weights_dense)

    torch.cuda.empty_cache()

    # =========================================================================
    # 4. FAZA KASKADOWA: SEGMENTACJA
    # =========================================================================
    log("\n--- ROZPOCZĘCIE TRENINGU: SEGMENTACJA KASKADOWA ---")
    
    # Loadery do segmentacji (korzystają z lokalnych transformacji modelu gęstości)
    dl_train_seg = dl_train_dense
    dl_val_seg   = dl_val_dense
    
    model_seg = SegResNet(
        spatial_dims=3,
        in_channels=1 + CONFIG["num_classes_combined"],
        out_channels=CONFIG["num_classes_seg"],
        init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], dropout_prob=0.2,
    ).to(DEVICE)
    
    seg_save_path = os.path.join(CONFIG["model_save_dir"], "best_metric_model_seg.pth")
    if os.path.exists(seg_save_path):
        model_seg.load_state_dict(torch.load(seg_save_path))
        
    optimizer_seg = torch.optim.AdamW(model_seg.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler_seg = ReduceLROnPlateau(optimizer_seg, mode="max", patience=5, factor=0.5)
    loss_function_seg = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, gamma=2.0)
    dsc_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    
    post_pred = AsDiscrete(argmax=True, to_onehot=CONFIG["num_classes_seg"])
    post_label = AsDiscrete(to_onehot=CONFIG["num_classes_seg"])
    scaler_seg = torch.amp.GradScaler(device='cuda')

    best_dsc, epoch_no_improve = -1.0, 0
    
    for epoch in range(CONFIG["epochs_seg"]):
        model_seg.train()
        epoch_loss, step = 0, 0
        
        for batch in dl_train_seg:
            step += 1
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            B, _, D, H, W = inputs.shape
            
            true_classes = batch["combined_id"].to(DEVICE).long().view(-1)
            if true_classes.shape[0] == 1 and B > 1:
                true_classes = true_classes.expand(B)
                
            class_channels = torch.zeros((B, CONFIG["num_classes_combined"], D, H, W), device=DEVICE, dtype=inputs.dtype)
            for b in range(B):
                # Teacher forcing z lekkim szumem (dropout 15%) dla warunkowania kaskadowego
                if random.random() > 0.15: 
                    class_channels[b, true_classes[b], ...] = 1.0
                    
            seg_inputs = torch.cat([inputs, class_channels], dim=1)
            
            optimizer_seg.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model_seg(seg_inputs)
                loss = loss_function_seg(outputs, labels)
            
            scaler_seg.scale(loss).backward()
            scaler_seg.step(optimizer_seg)
            scaler_seg.update()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            log(f"[Segmentacja] Epoka {epoch + 1} | Loss: {epoch_loss/step:.4f} | LR: {scheduler_seg.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 10 == 0:
            model_seg.eval()
            with torch.no_grad():
                for val_batch in dl_val_seg:
                    val_in, val_lbl = val_batch["image"].to(DEVICE), val_batch["label"].to(DEVICE)
                    
                    val_in_struct = F.interpolate(val_in, size=(CONFIG["global_size"],)*3, mode="trilinear")
                    struct_logits = model_struct(val_in_struct)
                    struct_pred = torch.argmax(struct_logits, dim=1)
                    
                    dense_logits = eval_density_sliding_window(val_in, val_lbl, model_dense, CONFIG["crop_size"])
                    dense_pred = torch.argmax(dense_logits, dim=1)
                    
                    combined_pred = (struct_pred * 2) + dense_pred
                    
                    B_val, _, D_val, H_val, W_val = val_in.shape
                    class_channels = torch.zeros((B_val, CONFIG["num_classes_combined"], D_val, H_val, W_val), device=DEVICE, dtype=val_in.dtype)
                    for b in range(B_val):
                        class_channels[b, combined_pred[b], ...] = 1.0
                        
                    seg_inputs = torch.cat([val_in, class_channels], dim=1)
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        val_outs = sliding_window_inference(
                            seg_inputs, roi_size=(CONFIG["crop_size"],)*3, 
                            sw_batch_size=4, predictor=model_seg, overlap=0.5, mode="gaussian"
                        )
                    
                    val_outs = [post_pred(i) for i in val_outs]
                    val_lbls = [post_label(i) for i in val_lbl]
                    dsc_metric(y_pred=val_outs, y=val_lbls)
                    
            metric_res, _ = dsc_metric.aggregate()
            dsc = metric_res.item()
            dsc_metric.reset()
            
            log(f" > Ewaluacja [Segmentacja]: DSC = {dsc:.4f}")
            scheduler_seg.step(dsc)
            
            if dsc > best_dsc:
                best_dsc = dsc
                torch.save(model_seg.state_dict(), seg_save_path)
                log(f" > Zapisano najlepszy model segmentacji (DSC: {best_dsc:.4f})")
                epoch_no_improve = 0
            else:
                epoch_no_improve += 10
                if epoch_no_improve >= 150:
                    log(" > Early stopping dla Segmentacji.")
                    break

    log(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ZAKOŃCZONO PROCES.")