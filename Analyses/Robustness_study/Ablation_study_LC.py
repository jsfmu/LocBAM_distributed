"""
# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (see LICENSE file or http://www.apache.org/licenses/LICENSE-2.0)
# This file includes components adapted from MONAI (Apache 2.0).
"""

# Evaluates model performance across 5 cross-validation folds for each defined input shift.
# For each shift value, the corresponding model for each fold is loaded and evaluated.
# The Dice scores are averaged over the 5 folds, and both the mean and standard deviation are computed.
# These statistics are saved for each shift in a JSON file, allowing analysis of model robustness under different input perturbations.
#
# Usage:
#   python Ablation_study_LC.py --data-dir /path/to/btcv --runs-dir /path/to/runs
#
#   --data-dir  folder with dataset_0.json ... dataset_4.json (cross-validation splits, MONAI decathlon format)
#   --runs-dir  folder with LR_HAN2_<fold>/UNet_final_HAN2_small_<fold>.pth (one trained model per fold)


import argparse
import json
import os
import sys
import tempfile

import monai.losses
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from monai.config import print_config
from monai.data import CacheDataset, ThreadDataLoader, decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AddCoordinateChannelsd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

# Make the repository root importable, so BasicUNet/LocBAM is found no matter where the script is run from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from BasicUNet.LocBAM.LocBAM_1D import BasicUnetLocBAMs2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice robustness of LocBAM models under shifted coordinate channels.")
    parser.add_argument("--data-dir", default="/vol/ciamspace/datasets/btcv/",
                        help="folder containing dataset_0.json ... dataset_4.json")
    parser.add_argument("--runs-dir", default="/u/home/hodo/Documents/runs",
                        help="folder containing LR_HAN2_<fold>/UNet_final_HAN2_small_<fold>.pth")
    parser.add_argument("--output", default=None,
                        help="results JSON path (default: <runs-dir>/Dice_p_shift_SP_HAN2.json)")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    print_config()
    # the train dataset and the cross-validation split configs are here
    # os.environ["MONAI_DATA_DIRECTORY"] = "/home/compai/code/data/nnunet_raw/nnunet_dataset/Dataset101_BTCV_abdomen/data/btcv/"

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    num_classes = 14  # 14 for btcv dataset, 2 for lung_dataset
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_val_transforms():
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
                ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),

                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=False),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                # add channel for correct scores here
                # AddCoordinateChannelsd(keys=["image"], spatial_dims=[0,1]),
                AddCoordinateChannelsd(keys=["image"], spatial_dims=[0, 1, 2]),
                # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),

                EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            ]
        )
        return val_transforms

    def build_dataset(fold_idx):
        data_dir = args.data_dir  # or btcv)backup for w BPREg scores
        # data_dir = "/vol/ciamspace/datasets/Dataset006_Lung/"
        split_json = "dataset_" + str(fold_idx) + ".json"

        datasets = os.path.join(data_dir, split_json)
        val_files = load_decathlon_datalist(datasets, True, "validation")
        cache_rate = 1.0

        val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_num=6, cache_rate=cache_rate, num_workers=4)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        # set_track_meta(True)
        return val_loader, val_ds

    def validation(epoch_iterator_val, patch_size, batch_size, shift):

        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                print(val_inputs.shape)
                val_inputs[:, -3:, :, :, :] = val_inputs[:, -3:, :, :, :] + shift
                val_inputs[:, -3:, :, :, :] = val_inputs[:, -3:, :, :, :] + shift
                print(val_inputs.shape)

                # mixed precision on GPU, full precision on CPU
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038bcbccc
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    max_sample_size = 24  # dataset size for BTCV, 50 for lung ct
    torch.backends.cudnn.benchmark = True
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, batch=True, include_background=False,
                                            squared_pred=True)
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    # dice_surface = SurfaceDiceMetric(class_thresholds = [1,1,1,1,1,1,1,1,1,1,1,1,1], include_background=False, reduction = "none")

    global_step = 0

    # Define shifts
    shifts = [0.00, 0.01, 0.05, 0.10, 0.25, 0.5, 1.0]

    # Dictionary to store the average dice values per shift
    average_dice_per_shift = {}

    for shift in shifts:
        dice_vals = []  # Store dice scores for 5 folds

        for fold_index in range(5):
            shift_tensor = torch.tensor(shift, device=device)  # Ensure shift is on the correct device

            # Define model path dynamically per fold
            # loading appropriate model path here ie, with folds ascending from 0 to 4
            model_path = os.path.join(args.runs_dir, f"LR_HAN2_{fold_index}", f"UNet_final_HAN2_small_{fold_index}.pth")

            # Load the model
            model = BasicUnetLocBAMs2(
                spatial_dims=3,
                in_channels=1,
                out_channels=14,
                features=(32, 32, 64, 128, 256, 32),
                dropout=0.1,
                LocBAM_params=None,
            )
            # model = BasicUNet(spatial_dims=3,
            #             in_channels=4,
            #             out_channels=14,
            #             features=(32, 32, 64, 128, 256, 32))

            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Load dataset for current fold
            # patch_size = (128, 128, 128)
            # effective_batch_size = 2
            patch_size = (32, 32, 32)
            effective_batch_size = 128

            val_loader, val_ds = build_dataset(fold_idx=fold_index)

            # Validate and compute dice score
            epoch_iterator_val = tqdm(val_loader, desc=f"Validate Fold {fold_index} (Shift={shift})", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val, patch_size, effective_batch_size, shift_tensor)

            print(f"Fold {fold_index}, Shift {shift}: Dice Score = {dice_val:.4f}")
            dice_vals.append(dice_val)

        # Compute mean and standard deviation
        avg_dice = float(np.mean(dice_vals))
        stdev = float(np.std(dice_vals))

        # Store as a dictionary for JSON compatibility
        average_dice_per_shift[str(shift)] = {"mean": avg_dice, "stdev": stdev}

        print(f"Average Dice Score for Shift {shift}: {avg_dice:.4f}, Std Dev: {stdev:.4f}")

    # Save all average dice scores in one JSON file
    output_file = args.output or os.path.join(args.runs_dir, "Dice_p_shift_SP_HAN2.json")
    with open(output_file, "w") as f:
        json.dump(average_dice_per_shift, f, indent=4)

    print(f"Results saved to {output_file}")
