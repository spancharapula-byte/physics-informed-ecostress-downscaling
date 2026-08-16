import h5py
import glob
import numpy as np

h5_files = glob.glob("data/raw/**/*.h5", recursive=True)

if not h5_files:
    print("No .h5 files found in data/raw!")
else:
    h5_path = h5_files[0]
    print(f"Inspecting 2D/3D Image Rasters in: {h5_path}\n")
    
    with h5py.File(h5_path, 'r') as f:
        found_grids = 0
        def find_2d_datasets(name, obj):
            global found_grids
            # Filter strictly for 2D or 3D image arrays (H x W)
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                found_grids += 1
                print(f"[{found_grids}] Raster Name: {name}")
                print(f"    ├── Shape: {obj.shape}")
                print(f"    ├── Data Type: {obj.dtype}")
                # Print scale factor and fill value if available
                attrs = dict(obj.attrs)
                scale = attrs.get('scale_factor', attrs.get('scale', 1.0))
                fill = attrs.get('_FillValue', attrs.get('FillValue', 'None'))
                units = attrs.get('units', 'None')
                print(f"    └── Scale Factor: {scale} | Fill Value: {fill} | Units: {units}\n")

        f.visititems(find_2d_datasets)