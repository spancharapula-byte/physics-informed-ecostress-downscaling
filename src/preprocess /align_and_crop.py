import os
import rioxarray
import xarray as xr
import numpy as np

def align_and_process(sister_path, ecostress_path, output_dir="data/interim"):
    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading datasets with rioxarray...")
    sister_ds = rioxarray.open_rasterio(sister_path)
    ecostress_ds = rioxarray.open_rasterio(ecostress_path)

    # Ensure CRS is defined
    target_crs = sister_ds.rio.crs
    print(f"2. Aligning to target CRS: {target_crs}")
    ecostress_ds = ecostress_ds.rio.reproject(target_crs)

    # 3. Calculate spatial intersection (bounding box overlap)
    minx = max(sister_ds.rio.bounds()[0], ecostress_ds.rio.bounds()[0])
    miny = max(sister_ds.rio.bounds()[1], ecostress_ds.rio.bounds()[1])
    maxx = min(sister_ds.rio.bounds()[2], ecostress_ds.rio.bounds()[2])
    maxy = min(sister_ds.rio.bounds()[3], ecostress_ds.rio.bounds()[3])

    print(f"3. Cropping to intersection: [{minx}, {miny}, {maxx}, {maxy}]")
    sister_clip = sister_ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    ecostress_clip = ecostress_ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    # 4. Feature Extraction: Calculate NDVI from SISTER (e.g., Red = Band 1, NIR = Band 2 depending on sensor)
    # Formula: (NIR - Red) / (NIR + Red)
    if sister_clip.shape[0] >= 2:
        nir = sister_clip[1].astype(float)
        red = sister_clip[0].astype(float)
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        ndvi_da = xr.DataArray(ndvi, coords=[sister_clip.y, sister_clip.x], dims=["y", "x"])
        ndvi_da.rio.write_crs(target_crs, inplace=True)
        ndvi_da.rio.to_raster(os.path.join(output_dir, "sister_ndvi_fine.tif"))
        print("Saved derived NDVI raster.")

    # 5. Save aligned raster files
    sister_clip.rio.to_raster(os.path.join(output_dir, "sister_aligned_fine.tif"))
    ecostress_clip.rio.to_raster(os.path.join(output_dir, "ecostress_aligned_coarse.tif"))
    print("Alignment and cropping complete.")

if __name__ == "__main__":
    # Update these paths to point to your specific raw files
    sister_sample = "data/raw/sister/sample_sister.tif"
    ecostress_sample = "data/raw/ecostress/sample_ecostress.tif"
    
    if os.path.exists(sister_sample) and os.path.exists(ecostress_sample):
        align_and_process(sister_sample, ecostress_sample)
    else:
        print("Please specify valid paths to your SISTER and ECOSTRESS files in data/raw/")
