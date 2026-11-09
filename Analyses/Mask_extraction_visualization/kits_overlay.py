"""Plot a CT scan with its segmentation mask overlaid in axial, coronal and sagittal views.

Used by Mask_visualizations.ipynb with KiTS23 cases (labels: 1 kidney, 2 tumor, 3 cyst).
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

KITS_CLASSES = {
    1: ("Kidney", "#ff9999"),
    2: ("Tumor", "#66b3ff"),
    3: ("Cyst", "#99ff99"),
}


def load_case(ct_path, mask_path):
    """Load CT and mask, reoriented to RAS: array axes are (left->right, back->front, feet->head)."""
    ct_nii = nib.as_closest_canonical(nib.load(ct_path))
    mask_nii = nib.as_closest_canonical(nib.load(mask_path))
    ct = ct_nii.get_fdata()
    mask = np.asarray(mask_nii.dataobj).astype(int)
    # Soft-tissue window (same range as mask_extraction.py), scaled to 0-1
    ct = (np.clip(ct, -175, 250) + 175) / 425
    return ct, mask, ct_nii.header.get_zooms()


def find_center(mask, focus="tumor"):
    """Voxel (x, y, z) to slice through: 'tumor', 'right_kidney' or 'left_kidney'."""
    if focus == "tumor":
        return np.argwhere(mask == 2).mean(axis=0).astype(int)
    kidney = np.argwhere(mask == 1)
    midline = kidney[:, 0].mean()
    # In RAS, larger x is further toward the patient's right
    side = kidney[kidney[:, 0] > midline] if focus == "right_kidney" else kidney[kidney[:, 0] <= midline]
    return side.mean(axis=0).astype(int)


def plot_overlay(ct_path, mask_path, out_pdf, focus="tumor", classes=KITS_CLASSES, alpha=0.5):
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 16

    ct, mask, (sx, sy, sz) = load_case(ct_path, mask_path)
    x, y, z = find_center(mask, focus)
    print(f"shape {ct.shape}, voxel size {sx:.2f}x{sy:.2f}x{sz:.2f} mm, slices x={x} y={y} z={z}")

    num_classes = max(classes)
    cmap = mcolors.ListedColormap(["none"] + [classes[i][1] for i in range(1, num_classes + 1)])
    norm = mcolors.BoundaryNorm(np.arange(num_classes + 2), cmap.N)

    # Transposed so up = head/front; left-right flipped so the patient's right
    # is on the image's left (radiology convention).
    views = {
        "Axial Slice": (ct[::-1, :, z].T, mask[::-1, :, z].T, sy / sx),
        "Coronal Slice": (ct[::-1, y, :].T, mask[::-1, y, :].T, sz / sx),
        "Sagittal Slice": (ct[x, :, :].T, mask[x, :, :].T, sz / sy),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, (ct_slice, mask_slice, aspect)) in zip(axes, views.items()):
        ax.imshow(ct_slice, cmap="gray", origin="lower", aspect=aspect)
        ax.imshow(np.ma.masked_equal(mask_slice, 0), cmap=cmap, norm=norm, alpha=alpha,
                  origin="lower", aspect=aspect, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    return fig
