import os
import torch
import matplotlib.pyplot as plt

def visualize_sample(dataset_path="data/processed/train_dataset.pt", sample_idx=10):
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return

    data = torch.load(dataset_path)
    inputs = data["inputs"]    
    targets = data["targets"]  

    print(f"Total available samples: {inputs.shape[0]}")
    sample_idx = min(sample_idx, inputs.shape[0] - 1)

    inp = inputs[sample_idx].numpy()
    tar = targets[sample_idx, 0].numpy()

    dem = inp[8]  

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(inp[0], cmap="viridis")
    axes[0].set_title(f"Sample #{sample_idx} - Optical Feature (Band 1)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(dem, cmap="terrain")
    axes[1].set_title(f"Sample #{sample_idx} - Elevation (DEM)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(tar, cmap="inferno")
    axes[2].set_title(f"Sample #{sample_idx} - ECOSTRESS LST (Target)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_img = "figures/sample_patch_check.png"
    plt.savefig(out_img, dpi=200)
    print(f"Visualization saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    visualize_sample()