import os
import torch
import numpy as np

def verify_dataset(data_dir="data/processed"):

    train_file = os.path.join(data_dir, "train_dataset.pt")
    val_file = os.path.join(data_dir, "val_dataset.pt")

    for name, path in [("TRAIN", train_file), ("VALIDATION", val_file)]:
        if not os.path.exists(path):
            print(f"[{name}] File missing: {path}")
            continue

        data = torch.load(path)
        x = data["inputs"]
        y = data["targets"]

        print(f"\n{name} SPLIT:")
        print(f"  ├── Inputs Tensor Shape : {x.shape} (dtype: {x.dtype})")
        print(f"  ├── Targets Tensor Shape: {y.shape} (dtype: {y.dtype})")

        nan_inputs = torch.isnan(x).sum().item()
        nan_targets = torch.isnan(y).sum().item()
        inf_inputs = torch.isinf(x).sum().item()
        inf_targets = torch.isinf(y).sum().item()

        print(f"  ├── NaN Check  : Inputs={nan_inputs}, Targets={nan_targets}")
        print(f"  ├── Inf Check  : Inputs={inf_inputs}, Targets={inf_targets}")

        print(f"  ├── LST Range  : Min={y.min().item():.2f} K, Max={y.max().item():.2f} K, Mean={y.mean().item():.2f} K")
        print(f"  └── DEM Range  : Min={x[:, 8].min().item():.2f} m, Max={x[:, 8].max().item():.2f} m")

        if nan_inputs == 0 and nan_targets == 0 and inf_inputs == 0 and inf_targets == 0:
            print(f"  [STATUS: PASSED] {name} split is clean Prithvi and Srijan.")
        else:
            print(f"  [STATUS: FAILED] {name} split contains NaNs/Infs.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    verify_dataset()