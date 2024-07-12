import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from tqdm.contrib.concurrent import process_map
from math import floor, ceil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int,
                    help="Year to process")
parser.add_argument("--path_data",
                    type=str,
                    help="Path to directory to download data.")
args = parser.parse_args()


warnings.filterwarnings("ignore")
os.chdir(f"{args.path_data}/CAMS_emissions")


# Latitude limits (from the PM2.5 dataset)
path_ = f"{args.path_data}/CAMS_emissions/interpolated_"
lats = [-55, 68]

year = args.year

while os.path.isfile(f"{path_}/{year}.parquet"):
    year += 1
    print(year)

# Emissions to process
ems_ = ["black-carbon", "carbon-dioxide", "carbon-monoxide",
        "nitrogen-oxides"]
# Read the emissions dataset
dsEm = xr.open_dataset("anthropogenic/all_emissions.nc")\
    .sel(lat=slice(lats[0], lats[1]))
dsEm = dsEm[ems_]
# Get the coordinates from this dataset
coords_ = [[lat, lon] for lat in tqdm(dsEm.lat.values) for lon
           in dsEm.lon.values]


def closest_node(node, nodes):
    """
    Finds the nearest coordinate to value from an array of coordinates
    Args:
        nodes: Array of coordinates [lat, lon]
        node: Coordinates of point to find the closest coordinate in array
    Returns:
        The closest coordinate to value from the array of coordinates
    """
    from scipy.spatial import distance

    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def getBU(year, path_=f"{args.path_data}/GHSL/builtup_volume"):
    from os import chdir, path
    from xarray import open_dataset
    from rioxarray import open_rasterio
    from glob import glob1
    from numpy import argmin
    import re

    chdir(path_)
    # List the population datasets (geotiffs)
    files = glob1(path_, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    # Read the dataset - If the reprojected netcdf is not yet computed, do it
    # and save it with the same filename but different extension
    if not path.isfile(files[years.index(year)].replace(".tif", ".nc")):
        ds = open_rasterio(files[years.index(year)])
        # Reproject to EPSG4326 and save as a netcdf
        ds = ds.rio.reproject("EPSG:4326")
        ds = ds.sel(band=1).drop("band").to_dataset(name="buildUp")
        if "spatial_ref" in ds.coords:
            ds = ds.drop("spatial_ref")
        ds = ds.rename({"x": "lon", "y": "lat"})
        # Save it to save time next time
        ds.to_netcdf(files[years.index(year)].replace(".tif", ".nc"),
                     encoding={"buildUp": {"dtype": "int16", "zlib": True}})
    # Read the netcdf file
    ds = open_dataset(files[years.index(year)].replace(".tif", ".nc"))
    return ds


def loadPop(year):
    from rioxarray import open_rasterio
    import os
    from glob import glob1
    from numpy import argmin
    import re

    # Change working directory to population dataset
    path = f"{args.path_data}/world_pop"
    os.chdir(path)
    # List the population datasets (geotiffs)
    files = glob1(path, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    filename = files[years.index(year)]
    if not os.path.isfile(filename.replace("tif", "nc")):
        # Read the corresponding dataset
        ds = open_rasterio(files[years.index(year)]).sel(band=1)\
            .drop(["band", "spatial_ref"])\
            .rename({"x": "lon", "y": "lat"})\
            .to_dataset(name="pop")
        ds.to_netcdf(filename.replace("tif", "nc"),
                     encoding={"pop": {"zlib": True,
                                       "complevel": 5}})
        del ds
    ds = xr.open_dataset(filename.replace("tif", "nc"))
    return ds


# Get the degree of urbanisation
dsBU_ = getBU(year)
# Get the population xarray
dsPop_ = loadPop(year)


def popDU(coords, dsPop_=dsPop_, dsDU_=dsBU_, grid_size=10):
    # Get the closest coordinate from the land use dataset
    dsPop = dsPop_.sel(lat=slice(ceil(coords[0]+1), floor(coords[0]-1)),
                       lon=slice(floor(coords[1]-1), ceil(coords[1]+1)))
    dsPop = xr.where(dsPop > 1e10, 0, dsPop)
    coordsPop = [[lat, lon] for lat in dsPop.lat.values for lon
                 in dsPop.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords, coordsPop)
    # Get the index of the lat and lon from ds_temp to the corresponding
    # grid cell in which station lies in
    y_idx = int(np.where(dsPop.lat.values == coordClose[0])[0])
    x_idx = int(np.where(dsPop.lon.values == coordClose[1])[0])
    # Get the lats and lons around the station with specified grid_size
    lats = dsPop.lat.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
    lons = dsPop.lon.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
    dsPop = dsPop.sel(lon=slice(lons.min(), lons.max()),
                      lat=slice(lats.max(), lats.min()))

    # Degree of Urbanisation
    # Get the closest coordinate from the land use dataset
    dsDU = dsDU_.sel(lat=slice(ceil(coords[0]+1), floor(coords[0]-1)),
                     lon=slice(floor(coords[1]-1), ceil(coords[1]+1)))
    coordsDU = [[lat, lon] for lat in dsDU.lat.values for lon
                in dsDU.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords, coordsDU)
    # Get the index of the lat and lon from ds_temp to the corresponding
    # grid cell in which station lies in
    y_idx = int(np.where(dsDU.lat.values == coordClose[0])[0])
    x_idx = int(np.where(dsDU.lon.values == coordClose[1])[0])
    # Get the lats and lons around the station with specified grid_size
    lats = dsDU.lat.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
    lons = dsDU.lon.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
    dsDU = dsDU.sel(lon=slice(lons.min(), lons.max()),
                    lat=slice(lats.max(), lats.min()))

    # Merge them
    ds_ = dsPop.merge(dsDU.interp(lon=dsPop.lon, lat=dsPop.lat,
                                  method="nearest"))
    # Convert to DataFrame
    df_ = ds_.to_dataframe().reset_index(drop=False)
    # Normalise the population and buildUp columns in the 0-1 range
    df_["pop"] = (df_["pop"] - df_["pop"].min()) / \
        (df_["pop"].max()-df_["pop"].min())
    df_["buildUp"] = (df_["buildUp"] - df_["buildUp"].min()) / \
        (df_["buildUp"].max()-df_["buildUp"].min())
    # In case they are all the same, give them the same weight
    # (1/total members)
    df_["pop"].fillna(1/df_.shape[0], inplace=True)
    df_["buildUp"].fillna(1/df_.shape[0], inplace=True)
    # Add them and divide by 2 to get the combined weight
    df_ = df_.assign(weight=0.5*(df_["pop"]+df_["buildUp"]))
    return df_


def weighEm(coord, ds=dsEm):
    # Emissions to process
    ems_ = ["black-carbon", "carbon-dioxide", "carbon-monoxide",
            "nitrogen-oxides"]

    # Get the emissions data
    emissions = ds.sel(lat=coord[0], lon=coord[1], time=f"{year}-12-31")
    # Get the weights (population and degree of urbanisation)
    df = popDU(coord)
    # Add the emissions to the dataframe
    for em_ in ems_:
        df[em_] = emissions[em_].item()
    # Weight them wrt to the population and degree of urbanisation
    for em_ in ems_:
        df[em_] = df[em_] * df["weight"]
    # Drop the weights
    df = df.drop(["pop", "buildUp", "weight"], axis=1)
    # Add the coarse coordinates as well
    df = df.assign(lat_=coord[0], lon_=coord[1])
    return df


path_ = f"{args.path_data}/CAMS_emissions/interpolated_"
os.makedirs(path_, exist_ok=True)

print(f"\n\n Year: {year}\n\n")

if not os.path.isfile(f"{path_}/{year}.parquet"):
    df = pd.concat(process_map(weighEm, coords_,
                               max_workers=128,
                               chunksize=len(coords_)//512))

    df.to_parquet(f"{path_}/{year}.parquet", index=False,
                  compression='zstd', compression_level=18)
else:
    print("Already processed. . . Skipping. . .")
