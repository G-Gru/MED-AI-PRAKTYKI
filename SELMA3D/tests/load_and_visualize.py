import matplotlib
import torch

if torch.backends.mps.is_available():
    matplotlib.use('macosx')
else:
    matplotlib.use('Qt5Agg')

import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

file = "blood_vessel_patchvolume_005.mha"
type_of_tissue = "contiguous"

raw_path = f"../data/SELMA3D2026_training_annotated/{type_of_tissue}_structures/raw/" + file
gt_path = f"../data/SELMA3D2026_training_annotated/{type_of_tissue}_structures/gt/" + file

raw_image = sitk.ReadImage(raw_path)
gt_image = sitk.ReadImage(gt_path)

raw_volume = sitk.GetArrayFromImage(raw_image)
gt_volume = sitk.GetArrayFromImage(gt_image)

fig, (ax_raw, ax_gt) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

initial_slice = raw_volume.shape[0] // 2
max_slice = raw_volume.shape[0] - 1

img_raw = ax_raw.imshow(raw_volume[initial_slice, :, :], cmap="gray")
ax_raw.set_title(f"Raw Image (Slice {initial_slice}/{max_slice})")
ax_raw.axis("off")

img_gt = ax_gt.imshow(gt_volume[initial_slice, :, :], cmap="gray")
ax_gt.set_title(f"Ground Truth (Slice {initial_slice}/{max_slice})")
ax_gt.axis("off")

slider_ax = plt.axes([0.25, 0.08, 0.5, 0.03])
slider = Slider(
    ax=slider_ax,
    label="Slice",
    valmin=0,
    valmax=max_slice,
    valinit=initial_slice,
    valstep=1,
)

# Callback
def update(val):
    idx = int(slider.val)
    img_raw.set_data(raw_volume[idx, :, :])
    ax_raw.set_title(f"Raw Image (Slice {idx}/{max_slice})")
    img_gt.set_data(gt_volume[idx, :, :])
    ax_gt.set_title(f"Ground Truth (Slice {idx}/{max_slice})")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()