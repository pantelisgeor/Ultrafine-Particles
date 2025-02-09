import fiona
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from geopandas import points_from_xy
import geopandas as gpd
import joblib
import warnings
import argparse
from joblib import Parallel, delayed
import gc

warnings.filterwarnings("ignore")

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str, help="Path to directory to download data.")
args = parser.parse_args()

# Configure paths and directories
os.makedirs(f"{args.path_data}/global_roads", exist_ok=True)
os.chdir(f"{args.path_data}/global_roads")

# Load road network data
path_db = "groads-v1-global-gdb/gROADS_v1.gdb"
roads = gpd.read_file(path_db, layer="Global_Roads")[["FCLASS", "geometry"]]
roads = roads.set_crs("EPSG:4326")
roads["FCLASS"] = roads["FCLASS"].astype("category")  # Optimize memory usage

# Load grid coordinates and generate grid points
coords = joblib.load("code/data/coords.joblib")
lat_grid, lon_grid = np.meshgrid(coords[0], coords[1], indexing="ij")
grid = pd.DataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()})


# ------------------------------------------------------------------------- #
def process_chunk(chunk, roads, buffer_radius):
    # Create geometries
    geometries = points_from_xy(chunk.lon, chunk.lat).buffer(buffer_radius, cap_style=3)
    points_gdf = gpd.GeoDataFrame(
        {"lat": chunk.lat, "lon": chunk.lon, "geometry": geometries}, crs=roads.crs
    )

    # Spatial filtering using bounding boxes
    minx, miny, maxx, maxy = points_gdf.total_bounds
    roads_subset = roads.cx[minx:maxx, miny:maxy]

    if not roads_subset.empty:
        # Perform spatial join
        sjoin = gpd.sjoin(roads_subset, points_gdf, how="inner", predicate="intersects")
        # Keep only necessary columns
        sjoin = sjoin[["FCLASS", "geometry", "lat", "lon"]]
        return sjoin
    return None


# Spatial join processing parameters
BUFFER_RADIUS = 0.00833333330001551 / 2  # Half of grid resolution
CHUNK_SIZE = 1000000  # Adjust based on system memory

# Split grid into chunks
grid_chunks = [
    grid.iloc[start_idx : start_idx + CHUNK_SIZE]
    for start_idx in range(0, len(grid), CHUNK_SIZE)
]

# Process chunks in parallel
joins_chunks = Parallel(n_jobs=30, backend="loky")(
    delayed(process_chunk)(chunk, roads, BUFFER_RADIUS) for chunk in grid_chunks
)

# Filter out None results and combine
joins_chunks = [chunk for chunk in joins_chunks if chunk is not None]
# Combine all the chunks
final_joins = pd.concat(joins_chunks, ignore_index=True)

# Combine and save results
if joins_chunks:
    final_joins = pd.concat(joins_chunks, ignore_index=True)
    final_joins.to_parquet("joins.parquet", index=False, compression="gzip")
else:
    pd.DataFrame().to_parquet("joins.parquet", index=False, compression="gzip")


# ------------------------------------------------------------------------- #
# Calculate the total number of occurrences of each grid cell
roads_total = (
    final_joins.groupby(["lat", "lon"], as_index=False)
    .size()
    .rename(columns={"size": "roads"})
)
gc.collect()

# Create a grid DataFrame from the original lat/lon coordinates
grid = pd.DataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()})

# Merge with the grid to ensure all points are included
grid_ = pd.merge(grid, roads_total, on=["lat", "lon"], how="left")
gc.collect()

# Convert to xarray using the original coordinate dimensions
grid_xr = grid_.set_index(["lat", "lon"]).to_xarray()
gc.collect()

# Fill NaN values with 0
grid_xr = grid_xr.fillna(0)

# Save to netCDF with compression
grid_xr.to_netcdf(
    "roads.nc",
    encoding={
        "roads": {"zlib": True, "complevel": 6, "dtype": int, "_FillValue": -9999}
    },
)
