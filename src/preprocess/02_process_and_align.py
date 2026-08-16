import os
import glob
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import rioxarray
import xarray as xr

def extract_ecostress_geotiff(h5_path, ref_tif_path, output_dir="data/interim"):
    os.makedirs(output_dir, exist_ok=True)
    print("1. Extracting LST and Cloud Mask from ECOSTRESS HDF5...")
    
    with h5py.File(h5_path, 'r') as f:
        grid_group = f["HDFEOS/GRIDS/ECO_L2G_LSTE_70m/Data Fields"]
        lst = grid_group["LST"][:].astype(np.float32)
        cloud = grid_group["cloud"][:].astype(np.uint8)
        
    print(f"   Raw LST matrix shape: {lst.shape}")
    
    with rasterio.open(ref_tif_path) as ref:
        crs = ref.crs
        bounds = ref.bounds  # [left, bottom, right, top]
        
    invalid_mask = (cloud > 0) | (lst < 200.0) | (lst > 380.0) | np.isnan(lst)
    lst_clean = lst.copy()
    lst_clean[invalid_mask] = np.nan
    
    lst_tif_path = os.path.join(output_dir, "ecostress_lst_raw.tif")
    transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, lst.shape[1], lst.shape[0])
    
    with rasterio.open(
        lst_tif_path,
        'w',
        driver='GTiff',
        height=lst.shape[0],
        width=lst.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(lst_clean, 1)
        
    print(f"   Saved raw LST GeoTIFF: {lst_tif_path}")
    return lst_tif_path


def align_all_layers(sentinel_path, dem_path, ecostress_tif_path, output_dir="data/interim"):
    print("\n2. Aligning Sentinel-2, DEM, and ECOSTRESS grids...")
    
    s2 = rioxarray.open_rasterio(sentinel_path)
    dem = rioxarray.open_rasterio(dem_path)
    eco = rioxarray.open_rasterio(ecostress_tif_path)
    
    dem_aligned = dem.rio.reproject_match(s2)
    
    eco_aligned = eco.rio.reproject_match(s2)
    
    combined_fine = np.concatenate([s2.values, dem_aligned.values], axis=0)
    
    fine_da = xr.DataArray(
        combined_fine,
        coords={"band": list(range(1, 10)), "y": s2.y, "x": s2.x},
        dims=["band", "y", "x"]
    )
    fine_da.rio.write_crs(s2.rio.crs, inplace=True)
    
    fine_out = os.path.join(output_dir, "fine_inputs_9bands.tif")
    coarse_out = os.path.join(output_dir, "target_lst_aligned.tif")
    
    fine_da.rio.to_raster(fine_out)
    eco_aligned.rio.to_raster(coarse_out)
    
    print(f"\n Preprocessing successful!")
    print(f"   └── Fine Features (9 bands: S2 + Elevation): {fine_out} | Shape: {combined_fine.shape}")
    print(f"   └── Target LST (Temperature): {coarse_out} | Shape: {eco_aligned.shape}")


if __name__ == "__main__":
    s2_files = glob.glob("data/raw/**/Sentinel2_Features_Aug12.tif", recursive=True)
    dem_files = glob.glob("data/raw/**/NASADEM_Elevation*.tif", recursive=True)
    h5_files = glob.glob("data/raw/**/*.h5", recursive=True)
    
    if not (s2_files and dem_files and h5_files):
        print("Error: Could not find one of the required input files in data/raw!")
        print(f"Found S2: {s2_files}")
        print(f"Found DEM: {dem_files}")
        print(f"Found H5: {h5_files}")
    else:
        s2_path = s2_files[0]
        dem_path = dem_files[0]
        h5_path = h5_files[0]
        
        eco_tif = extract_ecostress_geotiff(h5_path, s2_path)
        
        align_all_layers(s2_path, dem_path, eco_tif)