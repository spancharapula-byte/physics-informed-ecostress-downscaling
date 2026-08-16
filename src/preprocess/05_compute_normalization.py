import os
import json
import torch
import numpy as np

def compute_scaling_statistics(
    train_path="data/processed/train_dataset.pt",
    output_json="data/processed/normalization_stats.json"
):
    print("--- Computing Dataset Normalization Statistics ---")
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    data = torch.load(train_path)
    inputs = data["inputs"]    
    targets = data["targets"]  

    input_means = []
    input_stds = []
    for c in range(inputs.shape[1]):
        channel_data = inputs[:, c, :, :].float()
        mean_val = float(channel_data.mean())
        std_val = float(channel_data.std())
        input_means.append(mean_val)
        input_stds.append(std_val)

    target_mean = float(targets.float().mean())
    target_std = float(targets.float().std())

    stats = {
        "num_train_samples": int(inputs.shape[0]),
        "patch_size": [int(inputs.shape[2]), int(inputs.shape[3])],
        "input_channels": 9,
        "input_means": input_means,
        "input_stds": input_stds,
        "target_name": "Land_Surface_Temperature_K",
        "target_mean_K": target_mean,
        "target_std_K": target_std
    }

    with open(output_json, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved normalization parameters to: {output_json}")
    print(f"Target LST Training Mean: {target_mean:.2f} K | Std: {target_std:.2f} K")

if __name__ == "__main__":
    compute_scaling_statistics()