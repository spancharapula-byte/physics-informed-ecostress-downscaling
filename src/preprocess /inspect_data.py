import os
import glob
import rasterio
import h5py

def inspect_geotiff(file_path):
    print(f"\n--- Inspecting GeoTIFF: {os.path.basename(file_path)} ---")
    with rasterio.open(file_path) as src:
        print(f"Bands: {src.count}")
        print(f"Dimensions (Width x Height): {src.width} x {src.height}")
        print(f"Spatial Resolution (Pixel Size): {src.res}")
        print(f"Coordinate Reference System (CRS): {src.crs}")
        print(f"Bounding Box: {src.bounds}")
        print(f"NoData Value: {src.nodata}")

def inspect_h5(file_path):
    print(f"\n--- Inspecting HDF5/NetCDF: {os.path.basename(file_path)} ---")
    with h5py.File(file_path, 'r') as f:
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
        f.visititems(print_attrs)

if __name__ == "__main__":
    raw_dir = "data/raw"
    print(f"Scanning directory: {raw_dir}")
    
    tif_files = glob.glob(f"{raw_dir}/**/*.tif", recursive=True)
    h5_files = glob.glob(f"{raw_dir}/**/*.h5", recursive=True) + glob.glob(f"{raw_dir}/**/*.nc", recursive=True)

    for f in tif_files[:3]:
        inspect_geotiff(f)

    for f in h5_files[:3]:
        inspect_h5(f)
