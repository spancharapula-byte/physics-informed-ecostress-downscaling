import os
import numpy as np
import rasterio
import torch

def extract_patches(fine_path, coarse_path, output_dir="data/processed", patch_size=128, scale_factor=4):
    os.makedirs(output_dir, exist_ok=True)
    coarse_patch_size = patch_size // scale_factor

    with rasterio.open(fine_path) as src_f, rasterio.open(coarse_path) as src_c:
        fine_img = src_f.read().astype(np.float32)       # Shape: [C, H_fine, W_fine]
        coarse_img = src_c.read(1).astype(np.float32)    # Shape: [H_coarse, W_coarse]

    # Handle NaNs / Fill values
    fine_img = np.nan_to_num(fine_img, nan=0.0)
    coarse_img = np.nan_to_num(coarse_img, nan=0.0)

    # Standardize (Z-score normalization)
    for c in range(fine_img.shape[0]):
        mean_val, std_val = np.mean(fine_img[c]), np.std(fine_img[c]) + 1e-8
        fine_img[c] = (fine_img[c] - mean_val) / std_val

    c_mean, c_std = np.mean(coarse_img), np.std(coarse_img) + 1e-8
    coarse_img = (coarse_img - c_mean) / c_std

    _, H_f, W_f = fine_img.shape
    stride = patch_size // 2  # 50% overlap

    patches_fine = []
    patches_coarse = []

    for y in range(0, H_f - patch_size + 1, stride):
        for x in range(0, W_f - patch_size + 1, stride):
            f_patch = fine_img[:, y:y+patch_size, x:x+patch_size]
            
            # Corresponding coordinates in coarse image
            y_c, x_c = y // scale_factor, x // scale_factor
            c_patch = coarse_img[y_c:y_c+coarse_patch_size, x_c:x_c+coarse_patch_size]

            if c_patch.shape == (coarse_patch_size, coarse_patch_size):
                patches_fine.append(torch.from_numpy(f_patch))
                patches_coarse.append(torch.from_numpy(c_patch).unsqueeze(0))

    # Save extracted tensors
    dataset = {
        "fine_features": torch.stack(patches_fine),
        "coarse_lst": torch.stack(patches_coarse)
    }
    output_file = os.path.join(output_dir, "train_patches.pt")
    torch.save(dataset, output_file)
    print(f"Extracted and saved {len(patches_fine)} patches to {output_file}")

if __name__ == "__main__":
    fine_file = "data/interim/sister_aligned_fine.tif"
    coarse_file = "data/interim/ecostress_aligned_coarse.tif"

    if os.path.exists(fine_file) and os.path.exists(coarse_file):
        extract_patches(fine_file, coarse_file)
    else:
        print("Aligned files not found. Please run 02_align_and_crop.py first.")
