import os
import glob
import random
import torch
import numpy as np
import matplotlib

# 1. SET BACKEND BEFORE IMPORTING PYPLOT (Crucial for macOS)
if torch.backends.mps.is_available():
    matplotlib.use('macosx')
else:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm

from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    ToTensord,
    MapTransform,
    SpatialPadd
)
from monai.data import ITKReader
from monai.inferers import sliding_window_inference

# ==========================================
# 1. KONFIGURACJA
# ==========================================
MODEL_PATH = "../trained_models/best_metric_model.pth"
BASE_DIR = "../data/SELMA3D2026_training_annotated"
CROP_SIZE = 96
NUM_SAMPLES_PER_CLASS = 2


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


# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================
def get_files_for_category(category_name, class_id):
    raw_dir = os.path.join(BASE_DIR, category_name, "raw")
    gt_dir = os.path.join(BASE_DIR, category_name, "gt")
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.mha")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.mha")))
    return [{"image": r, "label": g, "class_id": class_id} for r, g in zip(raw_files, gt_files)]


contig_files = get_files_for_category("contiguous_structures", 1)
isolated_files = get_files_for_category("isolated_structures", 2)

random.seed()
selected_files = random.sample(contig_files, min(NUM_SAMPLES_PER_CLASS, len(contig_files))) + \
                 random.sample(isolated_files, min(NUM_SAMPLES_PER_CLASS, len(isolated_files)))
random.shuffle(selected_files)

eval_transforms = Compose([
    LoadImaged(keys=["image", "label"], reader=ITKReader),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiClassd(keys=["label"], class_key="class_id"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"]),
])

# ==========================================
# 3. ZAŁADOWANIE MODELU
# ==========================================
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
    print(f"Pomyślnie wczytano wagi z: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Nie znaleziono pliku modelu: {MODEL_PATH}")


# ==========================================
# 4. FUNKCJA WIZUALIZUJĄCA
# ==========================================
def show_slices(raw_img, gt_img, pred_img, window_title="Wizualizacja"):
    z_max = raw_img.shape[0] - 1
    z_init = z_max // 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.canvas.manager.set_window_title(window_title)
    plt.subplots_adjust(bottom=0.25)

    img1 = axes[0].imshow(raw_img[z_init, :, :], cmap='gray')
    axes[0].set_title("Oryginał (RAW)")
    axes[0].axis('off')

    img2 = axes[1].imshow(gt_img[z_init, :, :], cmap='viridis', interpolation='nearest', vmin=0, vmax=2)
    axes[1].set_title("Maska GT (Ekspercka)")
    axes[1].axis('off')

    img3 = axes[2].imshow(pred_img[z_init, :, :], cmap='viridis', interpolation='nearest', vmin=0, vmax=2)
    axes[2].set_title("Predykcja Modelu")
    axes[2].axis('off')

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.04])
    slider = Slider(
        ax=ax_slider,
        label='Przekrój (Oś Z)',
        valmin=0,
        valmax=z_max,
        valinit=z_init,
        valstep=1,
        valfmt='%d'
    )

    def update(val):
        z = int(slider.val)
        img1.set_data(raw_img[z, :, :])
        img2.set_data(gt_img[z, :, :])
        img3.set_data(pred_img[z, :, :])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print(f"\nWyświetlam: {window_title} (Zamknij okno wykresu, aby przejść do kolejnego)")
    plt.show()


# ==========================================
# 5. PRE-KOMPUTACJA (Osobno od renderingu GUI)
# ==========================================
processed_results = []

print("\nRozpoczynanie przetwarzania obrazów...")
with torch.no_grad():
    for file_dict in tqdm(selected_files, desc="Całkowity postęp plików"):
        file_name = os.path.basename(file_dict['image'])

        data = eval_transforms(file_dict)
        img_tensor = data["image"].unsqueeze(0).to(device)
        gt_tensor = data["label"].squeeze()

        roi_size = (CROP_SIZE, CROP_SIZE, CROP_SIZE)
        sw_batch_size = 4

        # progress=True dodaje pasek postępu dla kadrów 3D wewnątrz MONAI
        outputs = sliding_window_inference(
            img_tensor,
            roi_size,
            sw_batch_size,
            model,
            overlap=0.25,
            progress=True
        )

        pred_tensor = torch.argmax(outputs, dim=1).squeeze()

        raw_np = data["image"].squeeze().cpu().numpy().transpose(2, 1, 0)
        gt_np = gt_tensor.cpu().numpy().transpose(2, 1, 0)
        pred_np = pred_tensor.cpu().numpy().transpose(2, 1, 0)

        vmax_raw = np.percentile(raw_np, 99)
        raw_np = np.clip(raw_np, 0, vmax_raw)
        if vmax_raw > 0:
            raw_np = raw_np / vmax_raw

        processed_results.append((file_name, raw_np, gt_np, pred_np))

# ==========================================
# 6. RENDERING WIZUALIZACJI
# ==========================================
print("\nObliczenia zakończone. Otwieranie okien wizualizacji...")
for file_name, raw_np, gt_np, pred_np in processed_results:
    show_slices(raw_np, gt_np, pred_np, window_title=file_name)

print("\nWszystkie wizualizacje zostały zakończone.")