"""
This script generates the global data for a user defined
year on the 1x1 km grid for the UFP model inference step.
"""
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import gc
from tqdm.contrib.concurrent import process_map
import argparse
import glob
import re
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="Arguments to process data for UFP project")
parser.add_argument("--idx", help="Index of longitude arrays", type=int,
                    default=None)
parser.add_argument("--year", help="Year to process data for.", type=int,
                    default=None)
parser.add_argument("--path_data", type=str,
                    help="Path to directory where downloaded data is stored.")
parser.add_argument("--dest", type=str,
                    help="Path to save data created.")
args = parser.parse_args()

# Paths
path_dat = args.path_data
path_dest = args.dest

os.makedirs(path_dest, exist_ok=True)
os.chdir(path_dest)

# Read an NO2 dataset to use as a base
dsNO2 = xr.open_dataset(f"{path_dat}/NO2_1km/2010_final_1km.tif")
# Extract the longitudes and latitudes
lons = dsNO2.x.values
lats_ = dsNO2.y.values
del dsNO2
gc.collect()
# Split the lons into 50 equally sizes chunks
lons_ = np.array_split(lons, 50)


feats = ["land_1", "land_2", "land_3", "land_4", "land_5", "land_6",
         "land_7", "no2", "pop", "buildUp", "degreeUrb", "humanSettle",
         "pm25", "blackCarbon", "carbonDioxide", "carbonMonoxide",
         "nitrogenOxides"]


# ======================================================================= #
def get_coords(lons_, lats_, idx):
    """
    The functions gets an index for the split longitudes and
    constructs a lat/lon array with all of the combinations (grid)
    """
    coords = [[lat, lon] for lat in tqdm(lats_) for lon in lons_[idx]]
    gc.collect()
    return coords


def readLand(year, lons, lats,
             path=f"{path_dat}/land_use"):
    """
    Reads the raw land use data for a given year and subsets for the 
    given coordinates

    Args:
        year: Year to retrieve land use data for
        lons: Coordinates of strip to get data for from the
                corresponding grid 
        path: Directory where land use data is stored

    Returns:
        Land use dataset
    """
    from rioxarray import open_rasterio
    from glob import glob1
    from os import chdir
    import re

    # Change working directory to land use data directory
    chdir(path)
    # List the geotiff files in the directory
    files_ = glob1(path, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files_]
    year = min(years, key=lambda x: abs(x-year))
    # Read the corresponding dataset
    ds = open_rasterio(files_[years.index(year)]).sel(band=1)\
        .drop_vars(["band", "spatial_ref"])\
        .rename({"x": "lon", "y": "lat"})
    # Get the closest coordinate from the land use dataset
    ds_ = ds.sel(lon=slice(lons[0], lons[-1]), lat=slice(lats[0], lats[-1]))
    return ds_


def map_groups(data, groupings):
    """
    Maps the copernicus land use classes to a new classification system
    which groups the relevant classes together (from 23 classes to 7)

    Args:
        data: 2d numpy array (i.e. ds.values).
        groupings: dictionary with the new grouped classes.
    Returns:
        data: Mapped land use classes
    """
    import numpy as np

    data = np.stack(
        [np.stack(
            [data == int(x) for x in grouping])
         .sum(axis=0)
         .astype(bool) * i for i, grouping in groupings.items()]
    ).sum(axis=0)
    return data


def binaryLand(dsLand):
    """
    Create the binary land use masks
    """
    import xarray as xr
    import numpy as np

    lons = dsLand.lon.data
    lats = dsLand.lat.data
    dsLand = dsLand.to_array()
    dsLand_ = xr.Dataset({
        "land_1": xr.DataArray(
            data=np.where(dsLand == 1, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"}),
        "land_2": xr.DataArray(
            data=np.where(dsLand == 2, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"}),
        "land_3": xr.DataArray(
            data=np.where(dsLand == 3, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"}),
        "land_4": xr.DataArray(
            data=np.where(dsLand == 4, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"}),
        "land_5": xr.DataArray(
            data=np.where(dsLand == 5, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"}),
        "land_6": xr.DataArray(
            data=np.where(dsLand == 6, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"}),
        "land_7": xr.DataArray(
            data=np.where(dsLand == 7, 1, 0)[0, :, :],
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "Land Use Binary mask"})
    }
    )
    return dsLand_


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


def readNO2(year, lats_, lons_,
            path_dat_=f"{path_dat}/NO2_1km"):
    """
    Returns the NO2 data for the given year and longitudes/latitudes
    """
    from glob import glob1
    from pandas import DataFrame
    from numpy import argmin
    from rioxarray import open_rasterio
    from os import path
    from xarray import open_dataset

    # List the geotif files in the directory
    files = glob1(path_dat_, "*.tif")
    files = DataFrame({"year": [int(re.search(r"(\d{4})", x).group(1))
                                for x in files],
                       "filename": files})
    files = files.sort_values(by="year", ascending=True).reset_index(drop=True)
    # If the year is not in the list, load the closest year
    if year not in files.year.unique():
        year = files.year.values[argmin([abs(year-x) for x
                                         in files.year.values])]
    if path.isfile(f"{path_dat_}/{year}_final_1km.nc"):
        # Read the tif file corresponding to this year
        ds = open_dataset(f"{path_dat_}/{year}_final_1km.nc")\
            .sel(lon=slice(lons_[0], lons_[-1]),
                 lat=slice(lats_[0], lats_[-1]))
        return ds
    else:
        ds = open_rasterio(f"{path_dat_}/{year}_final_1km.tif")\
            .sel(band=1).drop_vars(["band", "spatial_ref"])\
            .rename({"x": "lon", "y": "lat"})\
            .to_dataset(name="no2")
        # Save it
        ds.to_netcdf(f"{path_dat_}/{year}_final_1km.nc",
                     encoding={"no2": {"zlib": True,
                                       "complevel": 5}})
        del ds
        ds = open_dataset(f"{path_dat_}/{year}_final_1km.nc")\
            .sel(lon=slice(lons_[0], lons_[-1]),
                 lat=slice(lats_[0], lats_[-1]))
        return ds


def readPM25(year, lats_, lons_,
             path_=f"{path_dat}/PM25"):
    """
    Read the PM2.5 dataset for the given year and indexes it
    wrt to the latitudes/longitudes specified.
    """
    from glob import glob1
    from pandas import DataFrame
    from xarray import open_dataset

    dat = glob1(path_, "*.nc")
    dat = DataFrame({
        "year": [int(x.split("-")[-1][:4]) for x in dat],
        "path": [f"{path_}/{x}" for x in dat]
    }).sort_values(by="year").reset_index(drop=True)
    # Load the netcdf for the year into an xarray
    ds = open_dataset(dat.loc[dat.year == year].path.item())\
        .sel(lat=slice(lats_[-1]+0.05, lats_[0]-0.05),
             lon=slice(lons_[0]-0.05, lons_[-1]+0.05))
    return ds


def readEmissions(year, lats_, lons_,
                  path_=f"{path_dat}/CAMS_emissions/interpolated_"):
    from xarray import open_dataset
    from glob import glob1
    import pandas as pd

    # List the netcdf files
    filesEm = glob1(path_, "*.nc")
    filesEm = pd.DataFrame({"year": [int(x.split(".")[0]) for x in filesEm],
                            "path": [f"{path_}/{x}" for x in filesEm]})\
        .sort_values(by="year", ascending=True)\
        .reset_index(drop=True)
    ds = open_dataset(filesEm.loc[filesEm.year == year].path.item())\
        .sel(lat=slice(lats_[-1], lats_[0]),
             lon=slice(lons_[0], lons_[-1]))
    return ds


def readPop(year, lats_, lons_,
            path_=f"{path_dat}/world_pop"):
    from xarray import open_dataset
    from glob import glob1
    from numpy import argmin
    import re

    # List the population datasets (geotiffs)
    files = glob1(path_, "*.nc")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    # Read the corresponding dataset
    ds = open_dataset(f"{path_}/{files[years.index(year)]}")\
        .sel(lat=slice(lats_[0], lats_[-1]), lon=slice(lons_[0], lons_[-1]))
    return ds


def readBuildUp(year, lats_, lons_,
                path_=f"{path_dat}/GHSL/builtup_volume"):
    from glob import glob1
    from numpy import argmin
    from xarray import open_dataset
    from rioxarray import open_rasterio
    from os import path
    import re

    # List the population datasets (geotiffs)
    files = glob1(path_, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    # Read the dataset
    if not path.isfile(
            f"{path_}/{files[years.index(year)].replace('.tif', '.nc')}"):
        ds = open_rasterio(f"{path_}/{files[years.index(year)]}")
        # Reproject to EPSG4326 and save as a netcdf
        ds = ds.rio.reproject("EPSG:4326")
        ds = ds.sel(band=1).drop("band").to_dataset(name="buildUp")
        ds = ds.rename({"x": "lon", "y": "lat"})
        # Save it to save time next time
        ds.to_netcdf(
            f"{path_}/{files[years.index(year)].replace('.tif', '.nc')}",
            encoding={"buildUp": {"dtype": "int32", "zlib": True}})
    ds = open_dataset(
        f"{path_}/{files[years.index(year)].replace('.tif', '.nc')}")\
        .sel(lat=slice(lats_[0], lats_[-1]), lon=slice(lons_[0], lons_[-1]))
    return ds


def readDegUrb(year, lats_, lons_,
               path_=f"{path_dat}/GHSL/degree_urbanisation"):
    from os import path
    from xarray import open_dataset
    from rioxarray import open_rasterio
    from glob import glob1
    from numpy import argmin
    import re

    # List the population datasets (geotiffs)
    files = glob1(path_, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    # Read the dataset - If the reprojected netcdf is not yet computed, do it
    # and save it with the same filename but different extension
    if not path.isfile(
            f"{path_}/{files[years.index(year)].replace('.tif', '.nc')}"):
        ds = open_rasterio(files[years.index(year)])
        # Reproject to EPSG4326 and save as a netcdf
        ds = ds.rio.reproject("EPSG:4326")
        ds = ds.sel(band=1).drop("band").to_dataset(name="degUrb")
        if "spatial_ref" in ds.coords:
            ds = ds.drop("spatial_ref")
        ds = ds.rename({"x": "lon", "y": "lat"})
        # Save it to save time next time
        ds.to_netcdf(
            f"{path_}/{files[years.index(year)].replace('.tif', '.nc')}",
            encoding={"degUrb": {"dtype": "int16", "zlib": True}})
    # Read the netcdf file
    ds = open_dataset(
        f"{path_}/{files[years.index(year)].replace('.tif', '.nc')}")\
        .sel(lat=slice(lats_[0], lats_[-1]), lon=slice(lons_[0], lons_[-1]))
    return ds


def readHuSet(year, lats_, lons_, idx,
              path_=f"{path_dat}/GHSL/human_settlement_wsg84"):
    from glob import glob1
    from numpy import argmin
    import re
    from xarray import open_dataset

    # List the population datasets (geotiffs)
    files = glob1(path_, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    if not os.path.isfile(f"{path_}/{files[years.index(year)]}"
                          .replace(".tif", f"_idx_{idx}.nc")):
        # Read the netcdf file
        ds = open_dataset(
            f"{path_}/{files[years.index(year)]}")\
            .sel(band=1).squeeze().drop_vars(["band", "spatial_ref"])\
            .rename({"x": "lon", "y": "lat", "band_data": "HumSet"})\
            .sel(lat=slice(lats_[0], lats_[-1]),
                 lon=slice(lons_[0], lons_[-1]))
        ds.to_netcdf(f"{path_}/{files[years.index(year)]}"
                     .replace(".tif", f"_idx_{idx}.nc"),
                     encoding={"HumSet": {"zlib": True,
                                          "complevel": 5}})
        del ds
        gc.collect()
        ds = open_dataset(f"{path_}/{files[years.index(year)]}"
                          .replace(".tif", f"_idx_{idx}.nc"))
    else:
        ds = open_dataset(f"{path_}/{files[years.index(year)]}"
                          .replace(".tif", f"_idx_{idx}.nc"))
    return ds


def adjust_longitude(dataset: xr.Dataset) -> xr.Dataset:
    """Swaps longitude coordinates from range (0, 360) to (-180, 180)
    Args:
        dataset (xr.Dataset): xarray Dataset
    Returns:
        xr.Dataset: xarray Dataset with swapped longitude dimensions
    """
    lon_name = "lon"  # whatever name is in the data

    # Adjust lon values to make sure they are within (-180, 180)
    dataset["_longitude_adjusted"] = xr.where(
        dataset[lon_name] > 180, dataset[lon_name] - 360, dataset[lon_name]
    )
    dataset = (
        dataset.swap_dims({lon_name: "_longitude_adjusted"})
        .sel(**{"_longitude_adjusted": sorted(dataset._longitude_adjusted)})
        .drop_vars(lon_name)
    )

    dataset = dataset.rename({"_longitude_adjusted": lon_name})
    return dataset


def read2t(year, lats_, lons_,
           path_=f"{path_dat}/ERA5/t2m_yearly.nc"):
    # Read the dataset into an xarray
    ds = xr.open_dataset(path_).rename({"longitude": "lon",
                                        "latitude": "lat"})
    ds = adjust_longitude(ds)

    ds = ds.sel(time=f"{year}-12-31", lat=slice(lats_[0], lats_[-1]),
                lon=slice(lons_[0], lons_[-1])).squeeze()\
        .drop_vars("time")
    gc.collect()
    return ds


# ======================================================================= #
# Needed
land_maps = pd.read_pickle(f"{path_dat}/land_use/copernicus_mappings.pickle")
# Destination directory
path_infer = f"{path_dest}/infer"
os.makedirs(path_infer, exist_ok=True)
# Other variables
# List the datasets
try:
    if not args.year:
        dats_ = glob.glob1(path_infer, "*.parquet")
        dats_ = pd.DataFrame({
            "year": [int(x.split("_")[0]) for x in dats_],
            "idx": [int(x.split("_")[-1].replace(".parquet", ""))
                    for x in dats_],
        }).sort_values(by=["year", "idx"]).reset_index(drop=True)
        year, idx = dats_.year.values[-1], dats_.idx.values[-1] + 1
        if idx == 50:
            year, idx = year + 1, 0
    else:
        year = args.year
        idx = args.idx
except IndexError:
    year, idx = 2010, 0

coords = get_coords(lons_, lats_, idx=idx)

print(f"\t-------------\n\tYear: {year} -- idx: {idx}\n\t-------------")


# ======================================================================= #
if not os.path.isfile(f"{path_dest}/land/{year}_{idx}.nc"):
    # Land use (Read the land use dataset for the specified longitudes)
    dsLand = readLand(year=year, lons=lons_[idx], lats=lats_)
    dsLand.values = map_groups(dsLand, land_maps)
    dsLand = dsLand.to_dataset(name="land")
    # Create the binary mask datasets
    dsLand = binaryLand(dsLand)
    gc.collect()
    # Save it compressed
    os.makedirs(f"{path_dest}/land", exist_ok=True)
    dsLand.to_netcdf(f"{path_dest}/land/{year}_{idx}.nc",
                     encoding={"land_1": {"dtype": int, "zlib": True,
                               "complevel": 6},
                               "land_2": {"dtype": int, "zlib": True,
                                          "complevel": 6},
                               "land_3": {"dtype": int, "zlib": True,
                                          "complevel": 6},
                               "land_4": {"dtype": int, "zlib": True,
                                          "complevel": 6},
                               "land_5": {"dtype": int, "zlib": True,
                                          "complevel": 6},
                               "land_6": {"dtype": int, "zlib": True,
                                          "complevel": 6},
                               "land_7": {"dtype": int, "zlib": True,
                                          "complevel": 6}})
else:
    dsLand = xr.open_dataset(f"{path_dest}/land/{year}_{idx}.nc")


def getLand(coords_, grid_size=10, dsLand=dsLand):
    """
    Function to return the land use for the specified grid point
    """
    # Get the closest coordinate from the land use dataset
    ds_ = dsLand.sel(lat=slice(coords_[0]+0.012, coords_[0]-0.012),
                     lon=slice(coords_[1]-0.012, coords_[1]+0.012))
    coordsLand = [[lat, lon] for lat in ds_.lat.values for lon
                  in ds_.lon.values]
    if len(coordsLand) == 0:
        df = pd.DataFrame({"year": [year]})
        for i in range(0, 7):
            df[f"land_{i+1}"] = None
            del i
        return df
    # Get the closest coordinate from the land use dataset
    try:
        coordClose = closest_node(coords_, coordsLand)
    except Exception as e:
        print(coords_, e)
        ds_temp = dsLand.sel(lat=coords_[0], lon=coords_[1],
                             method="nearest")
        coordClose = dsLand.lat.item(), dsLand.lon.item()
        del ds_temp
    # Get the index of the lat and lon from ds_temp to the corresponding
    # grid cell in which station lies in
    y_idx = np.where(ds_.lat.values == coordClose[0])[0].item()
    x_idx = np.where(ds_.lon.values == coordClose[1])[0].item()
    y_idx = 5 if y_idx < 5 else y_idx
    x_idx = 5 if x_idx < 5 else x_idx
    # Get the lats and lons around the station with specified grid_size
    lats = ds_.lat.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
    lons = ds_.lon.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
    # Select the coordinate sets above
    ds_ = ds_.sel(lon=slice(lons.min(), lons.max()),
                  lat=slice(lats.max(), lats.min()))
    # Calculate the percentage of each land class in the ds_ xarray
    # len(lats)*len(lons)  # Total number of grid cells
    totalCells = grid_size**2
    df = pd.DataFrame({"year": [year]})
    for i in range(0, 7):
        df[f"land_{i+1}"] = ds_[f"land_{i+1}"].sum().item() / totalCells
        del i
    return df


if not os.path.isfile(f"{path_dest}/land/{year}_{idx}.parquet"):
    print("\t-------------\n\tLand Use\n\t-------------")
    dfLand = pd.concat(process_map(getLand, coords, max_workers=20,
                                   chunksize=len(coords)//100))
    gc.collect()
    dfLand = dfLand.reset_index(drop=True)
    gc.collect()
    dfLand = pd.concat([pd.DataFrame(coords, columns=["lat", "lon"]),
                        dfLand], axis=1)
    gc.collect()
    dfLand.to_parquet(f"{path_dest}/land/{year}_{idx}.parquet",
                      compression="gzip", index=False)
    gc.collect()
else:
    dfLand = pd.read_parquet(f"{path_dest}/land/{year}_{idx}.parquet")


# ======================================================================= #
# --------------------------------- NO2 --------------------------------- #
print("\t-------------\n\tNO2\n\t-------------")
# Read the NO2 dataset
dsNO2 = readNO2(year, lons_=lons_[idx], lats_=lats_)


def getNO2(coords_, ds=dsNO2):
    """
    Function to read the 1km NO2 for a given year
    and return the data for a specified coordinate

    Args:
        year: The year to get data for
        coords: The coordinates to get the corresponding grid box
                ([lat, lon] form)
    Returns:
        no2: Annual NO2 average (float)
    """
    try:
        # Get the closest coordinates from the NO2 dataset
        ds_ = ds.sel(lon=slice(coords_[1] - 0.05, coords_[1] + 0.05),
                     lat=slice(coords_[0] + 0.05, coords_[0] - 0.05))
        coordsNO2 = [[lat, lon] for lat in ds_.lat.values for
                     lon in ds_.lon.values]
        # Get the closest coordinate from the NO2 dataset
        coordClose = closest_node(coords_, coordsNO2)
        # Get the NO2 concentration for the coordClose coordinate
        ds_ = ds_.sel(lon=coordClose[1], lat=coordClose[0])
        return ds_.no2.item()
    except Exception as e:
        print(coords_, "\n", e)
        return None


# Apply the function to all the coordinates
no2 = process_map(getNO2, coords, max_workers=128,
                  chunksize=len(coords)//512)
# Add it to the inital dataframe
dfLand = dfLand.assign(no2=no2)
del no2, getNO2, dsNO2
gc.collect()


# ======================================================================= #
# ------------------------------- PM2.5 --------------------------------- #
print("\t-------------\n\tPM2.5\n\t-------------")

dsPM25 = readPM25(year, lats_=lats_, lons_=lons_[idx])


def getPM25(coords_, dsPM25=dsPM25):
    # Get the closest coordinate from the PM2.5 dataset
    ds_ = dsPM25.sel(lat=slice(coords_[0]-0.05, coords_[0]+0.05),
                     lon=slice(coords_[1]-0.05, coords_[1]+0.05))
    coordsPM25 = [[lat, lon] for lat in ds_.lat.values for lon
                  in ds_.lon.values]
    if len(coordsPM25) == 0:
        return np.nan
    try:
        # Get the closest coordinate from the PM2.5 dataset
        coordClose = closest_node(coords_, coordsPM25)
        # Get the data for the corresponding grid cell
        ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
        return ds_.GWRPM25.item()
    except Exception:
        return np.nan


# Apply the function to get PM2.5 to all the coordinates
pm25 = process_map(getPM25, coords, max_workers=128,
                   chunksize=len(coords)//512)
# Add it to the initial dataframe
dfLand = dfLand.assign(pm25=pm25)
del pm25, getPM25, dsPM25
gc.collect()


# ======================================================================= #
# ------------------------------ Emissions ------------------------------ #
print("\t-------------\n\tEmissions\n\t-------------")
dsEm = readEmissions(year, lats_, lons_[idx])


def getEmissions(coords_, dsEm=dsEm):
    # Subset for the specified coordinates etc. . .
    try:
        # Read the emissions dataset for the year
        ds_ = dsEm.sel(lat=slice(coords_[0]-0.02, coords_[0]+0.02),
                       lon=slice(coords_[1]-0.02, coords_[1]+0.02))
        coordsEm_ = [[lat, lon] for lat in ds_.lat.values for lon
                     in ds_.lon.values]
        # Get the closest coordinate from the land use dataset
        coordClose = closest_node(coords_, coordsEm_)
        # Get the data for the corresponding grid cell
        ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
        # Put the data to return in a list
        dat_ret = list(ds_[["black-carbon", "carbon-dioxide",
                            "carbon-monoxide", "nitrogen-oxides"]]
                       .to_array().values)
    except Exception:
        dat_ret = [np.nan, np.nan, np.nan, np.nan]
    return dat_ret


# Apply the function to get the emissions for all the coordinates
# dfEm = pd.concat(process_map(getEmissions, coords, max_workers=85,
#                              chunksize=len(coords)//260))
dfEm = pd.DataFrame(process_map(getEmissions, coords, max_workers=128,
                                chunksize=len(coords)//512),
                    columns=["blackCarbon", "carbonDioxide",
                             "carbonMonoxide",
                             "nitrogenOxides"])
# Add it to the dataframe
dfLand = pd.concat([dfLand, dfEm.reset_index(drop=True)], axis=1)
del dfEm, getEmissions, dsEm
gc.collect()


# ======================================================================= #
# ----------------------------- Population ------------------------------ #
print("\t-------------\n\tPopulation\n\t-------------")
# Read the population dataset
dsPop = readPop(year=year, lats_=lats_, lons_=lons_[idx])


def getPop(coords_, dsPop=dsPop):
    try:
        # Get the closest coordinate from the land use dataset
        ds_ = dsPop.sel(lat=slice(coords_[0]+0.02, coords_[0]-0.02),
                        lon=slice(coords_[1]-0.02, coords_[1]+0.02))
        coordsPop = [[lat, lon] for lat in ds_.lat.values for lon
                     in ds_.lon.values]
        # Get the closest coordinate from the land use dataset
        coordClose = closest_node(coords_, coordsPop)
        # Get the data for the corresponding grid cell
        ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
        return ds_.pop.item()
    except Exception as e:
        print(coords_, "\n", e)
        return np.nan


# Apply the function to get the population for all the grid points
pop = process_map(getPop, coords, max_workers=128,
                  chunksize=len(coords)//512)
# Add it to the dataframe
dfLand = dfLand.assign(pop=pop)
del pop, getPop, dsPop


# ======================================================================= #
# --------------------------- Build Up Volume --------------------------- #
print("\t-------------\n\tBuildUp Volume\n\t-------------")
dsBU = readBuildUp(year=year, lats_=lats_, lons_=lons_[idx])


def buildVolume(coords_, dsBU=dsBU):
    # Get the closest coordinate from the land use dataset
    ds_ = dsBU.sel(lat=slice(coords_[0]+0.02, coords_[0]-0.02),
                   lon=slice(coords_[1]-0.02, coords_[1]+0.02))
    coordsBuildUp = [[lat, lon] for lat in ds_.lat.values for lon
                     in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords_, coordsBuildUp)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.buildUp.item()


# Apply the function to get the population for all the grid points
buildUp = process_map(buildVolume, coords, max_workers=128,
                      chunksize=len(coords)//512)
# Add it to the dataframe
dfLand = dfLand.assign(buildUp=buildUp)
del dsBU, buildVolume, buildUp


# ======================================================================= #
# ----------------------- Degree of Urbanisation ------------------------ #
print("\t-------------\n\tDegree of Urbanisation\n\t-------------")
dsDegUr = readDegUrb(year=year, lats_=lats_, lons_=lons_[idx])


def degreeUrban(coords_, dsDegUr=dsDegUr):
    # Get the closest coordinate from the land use dataset
    ds_ = dsDegUr.sel(lat=slice(coords_[0]+0.03, coords_[0]-0.03),
                      lon=slice(coords_[1]-0.03, coords_[1]+0.03))
    coordsDegreeUrban = [[lat, lon] for lat in ds_.lat.values for lon
                         in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords_, coordsDegreeUrban)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.degUrb.item()


# Apply the function to get the population for all the grid points
degUrb = process_map(degreeUrban, coords, max_workers=128,
                     chunksize=len(coords)//512)
# Add it to the dataframe
dfLand = dfLand.assign(degreeUrb=degUrb)
del degUrb, degreeUrban, dsDegUr


# ======================================================================= #
# --------------------------- Human Settlement -------------------------- #
print("\t-------------\n\tHuman Settlement\n\t-------------")
dsHS = readHuSet(year=year, lats_=lats_, lons_=lons_[idx], idx=idx)


def humanSettle(coords_, dsHS=dsHS):
    # Get the closest coordinate from the land use dataset
    ds_ = dsHS.sel(lat=slice(coords_[0]+0.02, coords_[0]-0.02),
                   lon=slice(coords_[1]-0.02, coords_[1]+0.02))
    coordsDegreeUrban = [[lat, lon] for lat in ds_.lat.values for lon
                         in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords_, coordsDegreeUrban)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.HumSet.item()


# Apply the function to get the population for all the grid points
HumSet = process_map(humanSettle, coords, max_workers=128,
                     chunksize=len(coords)//512)
# Add it to the dataframe
dfLand = dfLand.assign(humanSettle=HumSet)
del dsHS, HumSet, humanSettle
gc.collect()

# ======================================================================= #
# --------------------------- 2m Temperature --------------------------- #
print("\t-------------\n\t2m Temperature\n\t-------------")
ds2t = read2t(year, lats_, lons_=lons_[idx])


def get2t(coords_, ds2t=ds2t):
    # Get the data for the corresponding grid cell
    return ds2t.sel(lat=coords_[0], lon=coords_[1],
                    method="nearest").t2m.item()


# Apply the function to get the temperature dataset
# for all the grid points
t2m = process_map(get2t, coords, max_workers=128,
                  chunksize=len(coords)//512)
# Add it to the dataframe
dfLand = dfLand.assign(t2m=t2m)
del ds2t, get2t, t2m
gc.collect()


# Save it
dfLand.to_parquet(f"{path_infer}/{year}_{idx}.parquet",
                  compression="gzip", index=False)
