import os
import rasterio
import numpy as np
import torch
from tqdm import tqdm

def create_training_patches(
    fine_tif="data/interim/fine_inputs_9bands.tif",
    target_tif="data/interim/target_lst_aligned.tif",
    output_dir="data/processed",
    patch_size=128,
    stride=64,             # 50% overlap for data augmentation
    max_nan_fraction=0.05  # Reject patches with > 5% missing/cloud pixels
):
    os.makedirs(output_dir, exist_ok=True)
    print("1. Reading aligned GeoTIFFs into memory...")

    with rasterio.open(fine_tif) as src:
        fine_data = src.read().astype(np.float32)  # Shape: [9, 3005, 3727]

    with rasterio.open(target_tif) as src:
        target_data = src.read(1).astype(np.float32)  # Shape: [3005, 3727]

    _, H, W = fine_data.shape
    print(f"   Scene dimensions: Height={H}, Width={W}")

    fine_patches = []
    target_patches = []

    print(f"2. Slicing into {patch_size}x{patch_size} patches with stride {stride}...")
    
    total_steps = len(range(0, H - patch_size + 1, stride)) * len(range(0, W - patch_size + 1, stride))
    
    with tqdm(total=total_steps, desc="Extracting Patches") as pbar:
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                f_patch = fine_data[:, y:y+patch_size, x:x+patch_size]
                t_patch = target_data[y:y+patch_size, x:x+patch_size]

                nan_ratio = (np.isnan(t_patch) | np.isnan(f_patch).any(axis=0)).mean()

                if nan_ratio <= max_nan_fraction:
                    f_patch = np.nan_to_num(f_patch, nan=0.0)
                    t_patch = np.nan_to_num(t_patch, nan=np.nanmedian(t_patch) if not np.isnan(np.nanmedian(t_patch)) else 300.0)

                    fine_patches.append(f_patch)
                    target_patches.append(t_patch[np.newaxis, :, :])  # Shape: [1, patch_size, patch_size]

                pbar.update(1)

    print(f"Valid patches retained: {len(fine_patches)} / {total_steps}")

    X = torch.from_numpy(np.stack(fine_patches)).float()    # Shape: [N, 9, 128, 128]
    Y = torch.from_numpy(np.stack(target_patches)).float()  # Shape: [N, 1, 128, 128]

    num_samples = X.shape[0]
    indices = torch.randperm(num_samples)
    split_idx = int(0.8 * num_samples)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_data = {"inputs": X[train_idx], "targets": Y[train_idx]}
    val_data = {"inputs": X[val_idx], "targets": Y[val_idx]}

    train_file = os.path.join(output_dir, "train_dataset.pt")
    val_file = os.path.join(output_dir, "val_dataset.pt")

    torch.save(train_data, train_file)
    torch.save(val_data, val_file)

    print("\nDataset ready for Training @Prithvi and @Srijan!")
    print(f"   └── Training set: {train_file} | Inputs: {train_data['inputs'].shape}, Targets: {train_data['targets'].shape}")
    print(f"   └── Validation set: {val_file} | Inputs: {val_data['inputs'].shape}, Targets: {val_data['targets'].shape}")


if __name__ == "__main__":
    create_training_patches()