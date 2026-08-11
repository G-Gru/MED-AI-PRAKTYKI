import os
import copy
import torch
import numpy as np
import tifffile
from tqdm import tqdm

from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    ToTensord,
    SpatialPadd
)
from monai.data import ITKReader
from monai.inferers import sliding_window_inference

# ==========================================
# 1. KONFIGURACJA
# ==========================================
MODEL_PATH = "../models/best_metric_model.pth"
HARDCODED_IMAGE_PATH = "../data/SELMA3D2026_training_annotated/contiguous_structures/raw/blood_vessel_patchvolume_005.mha"
OUTPUT_DIR = "../results"
CROP_SIZE = 96

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = BasicUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    features=(32, 32, 64, 128, 256, 32)
).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Załadowano wagi modelu z: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Nie znaleziono pliku modelu: {MODEL_PATH}")

# ==========================================
# 2. PIPELINE TRANSFORMACJI
# ==========================================
transform_geometry = Compose([
    LoadImaged(keys=["image"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    SpatialPadd(keys=["image"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
])

transform_model = Compose([
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image"]),
])

print(f"Przetwarzanie obrazu: {HARDCODED_IMAGE_PATH}")
data_raw = {"image": HARDCODED_IMAGE_PATH}
data_geom = transform_geometry(data_raw)
data_model = transform_model(copy.deepcopy(data_geom))
img_tensor = data_model["image"].unsqueeze(0).to(device)

# ==========================================
# 3. PREDYKCJA
# ==========================================
roi_size = (CROP_SIZE, CROP_SIZE, CROP_SIZE)

with torch.no_grad():
    outputs = sliding_window_inference(
        img_tensor,
        roi_size,
        sw_batch_size=4,
        predictor=model,
        overlap=0.25,
        progress=True
    )
    pred_tensor = torch.argmax(outputs, dim=1).squeeze()

raw_np = data_geom["image"].squeeze().numpy().transpose(2, 1, 0)
pred_np = pred_tensor.cpu().numpy().transpose(2, 1, 0)

# ==========================================
# 4. KOREKCJA INTENSYWNOŚCI
# ==========================================
vmin_raw = np.percentile(raw_np, 1)
vmax_raw = np.percentile(raw_np, 99)

raw_np = np.clip(raw_np, vmin_raw, vmax_raw)
raw_np = ((raw_np - vmin_raw) / (vmax_raw - vmin_raw + 1e-8)) * 255.0
raw_np = raw_np.astype(np.uint8)

# ==========================================
# 5. EKSPORT DLA ACTO3D (Fiji/ImageJ Multipage TIFF)
# ==========================================
# ACTO3D oczekuje pojedynczego pliku ułożonego w format (Z, C, Y, X)
num_slices, height, width = raw_np.shape
print(f"Tworzenie stosu Z-stack ImageJ dla ACTO3D w folderze '{OUTPUT_DIR}'...")

base_name = os.path.splitext(os.path.basename(HARDCODED_IMAGE_PATH))[0]
out_filepath = os.path.join(OUTPUT_DIR, f"{base_name}_stack.tif")

# Inicjalizacja macierzy: (Głębokość Z, 3 Kanały, Wysokość, Szerokość)
stack = np.zeros((num_slices, 3, height, width), dtype=np.uint8)

# Kanał 0: Oryginalny Skan
stack[:, 0, :, :] = raw_np
# Kanał 1: Maska kategorii 1 (Przemnożone do białego, aby ACTO3D poprawnie widziało obszar)
stack[:, 1, :, :] = (pred_np == 1).astype(np.uint8) * 255
# Kanał 2: Maska kategorii 2
stack[:, 2, :, :] = (pred_np == 2).astype(np.uint8) * 255

# Zapis przy użyciu standardu ImageJ
tifffile.imwrite(
    out_filepath,
    stack,
    imagej=True,
    resolution=(1.0, 1.0),
    metadata={
        'spacing': 1.0,         # Dystans fizyczny między klatkami na osi Z (rozwiązuje błąd odczytu)
        'unit': 'mm',           # Fizyczna jednostka (wymagana przez czytnik metadanych)
        'axes': 'ZCYX',         # Oznaczenie wymiarów: Z-stack, Kanały, Y, X (rozwiązuje problem z czwartym kanałem)
        'Labels': ['Original MRI', 'Mask Class 1', 'Mask Class 2'] # Tagi czytelne w Acto3D / Fiji
    }
)

print(f"Gotowe! Plik zapisany w: {os.path.abspath(out_filepath)}")
print("W ACTO3D użyj opcji: Open images -> Open ImageJ / Fiji TIFF")