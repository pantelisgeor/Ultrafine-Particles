import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str,
                    help="Path to directory where data is stored.")
parser.add_argument("--path_obs", type=str,
                    help="Path to csv that holds the" +
                    " ground station measurements.")
args = parser.parse_args()

pathDat = args.path_data


# -------------------- FUNCTION DEFINITIONS -------------------- #
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


def getNO2(year, coords, path_data=pathDat):
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
    from rioxarray import open_rasterio
    from math import floor, ceil
    from glob import glob1
    from pandas import DataFrame
    from numpy import argmin

    # Path to dataset directory
    path_dat = f"{path_data}/NO2_1km"
    # List the geotif files in the directory
    files = glob1(path_dat, "*.tif")
    files = DataFrame({"year": [int(x.split("_")[0]) for x in files],
                       "filename": files})
    files = files.sort_values(by="year", ascending=True).reset_index(drop=True)
    # If the year is not in the list, load the closest year
    if year not in files.year.unique():
        year = files.year.values[argmin([abs(year-x) for x
                                         in files.year.values])]
    # Read the tif file corresponding to this year
    ds = open_rasterio(f"{path_dat}/{year}_final_1km.tif")\
        .sel(band=1).drop_vars(["band", "spatial_ref"])
    # Get the closest coordinates from the NO2 dataset
    ds_ = ds.sel(x=slice(floor(coords[1] - 0.2), ceil(coords[1] + 0.2)),
                 y=slice(ceil(coords[0] + 0.2), floor(coords[0] - 0.2)))
    coordsNO2 = [[lat, lon] for lat in ds_.y.values for lon in ds_.x.values]
    # Get the closest coordinate from the NO2 dataset
    coordClose = closest_node(coords, coordsNO2)
    # Get the NO2 concentration for the coordClose coordinate
    ds_ = ds_.sel(x=coordClose[1], y=coordClose[0])
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
    data = np.stack(
        [np.stack(
            [data == int(x) for x in grouping])
         .sum(axis=0)
         .astype(bool) * i for i, grouping in groupings.items()]
    ).sum(axis=0)
    return data


def getLand(year, coords, grid_size=10,
            path=f"{pathDat}/land_use"):
    """
    Returns the land use data for the UFP regression model

    Args:
        year: Year to retrieve land use data for
        coords: Coordinates of point to get data for from the
                corresponding grid
        grid_size: Number of grid cell points to retrieve for land use dataset
                   (square around centre point denoted by coords)
        path: Directory where land use data is stored

    Returns:
        Land use dataset (percentage of each land use class in the grid box)
    """
    from rioxarray import open_rasterio
    from glob import glob1
    from os import chdir
    from math import floor, ceil
    from numpy import where, argmin
    from pandas import read_pickle, DataFrame
    import re

    # Change working directory to land use data directory
    chdir(path)
    year_ = year
    # List the geotiff files in the directory
    files_ = glob1(path, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files_]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    # Read the corresponding dataset
    ds = open_rasterio(files_[years.index(year)]).sel(band=1)\
        .drop_vars(["band", "spatial_ref"])\
        .rename({"x": "lon", "y": "lat"})
    # Get the closest coordinate from the land use dataset
    ds_ = ds.sel(lat=slice(ceil(coords[0]+1), floor(coords[0]-1)),
                 lon=slice(floor(coords[1]-1), ceil(coords[1]+1)))
    coordsUFP = [[lat, lon] for lat in ds_.lat.values for lon
                 in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords, coordsUFP)
    # Get the index of the lat and lon from ds_temp to the corresponding
    # grid cell in which station lies in
    y_idx = int(where(ds_.lat.values == coordClose[0])[0])
    x_idx = int(where(ds_.lon.values == coordClose[1])[0])
    # Get the lats and lons around the station with specified grid_size
    lats = ds_.lat.values[int(y_idx-(grid_size/2)):int(y_idx+(grid_size/2))]
    lons = ds_.lon.values[int(x_idx-(grid_size/2)):int(x_idx+(grid_size/2))]
    # Select the coordinate sets above
    ds_ = ds_.sel(lon=slice(lons.min(), lons.max()),
                  lat=slice(lats.max(), lats.min()))
    # Read the land use mappings
    landMaps = read_pickle("copernicus_mappings.pickle")
    # Map the land use classes to the semantically similar groups
    ds_.values = map_groups(ds_, landMaps)
    # Count the total number of mapped land classes in the ds_ xarray
    landCounts = [sum(sum(ds_.values == i)) for i in range(1, 8)]
    # Calculate the percentage of each land class in the ds_ xarray
    # len(lats)*len(lons)  # Total number of grid cells
    totalCells = grid_size**2
    df = DataFrame({"year": [year_]})
    for i, cnt in enumerate(landCounts):
        df[f"land_{i+1}"] = cnt / totalCells
        del i, cnt
    return df


def getPop(year, coords,
           path=f"{pathDat}/world_pop"):
    """
    Returns the population data for the UFP regression model

    Args:
        year: Year to retrieve land use data for
        coords: Coordinates of point to get data for from the
                corresponding grid
        path: Directory where world pop population data is stored

    Returns:
        Population data for the corresponding grid box where point lies in
    """
    from rioxarray import open_rasterio
    from os import chdir
    from glob import glob1
    from math import floor, ceil
    from numpy import argmin

    # Change working directory to population dataset
    chdir(path)
    # List the population datasets (geotiffs)
    files = glob1(path, "*.tif")
    years = [int(re.findall(r'\d{4}', x)[0]) for x in files]
    # If the year is not in the list, load the closest year
    if year not in years:
        year = years[argmin([abs(year-x) for x in years])]
    # Read the corresponding dataset
    ds = open_rasterio(files[years.index(year)]).sel(band=1)\
        .drop_vars(["band", "spatial_ref"])\
        .rename({"x": "lon", "y": "lat"})
    # Get the closest coordinate from the land use dataset
    ds_ = ds.sel(lat=slice(ceil(coords[0]+1), floor(coords[0]-1)),
                 lon=slice(floor(coords[1]-1), ceil(coords[1]+1)))
    coordsUFP = [[lat, lon] for lat in ds_.lat.values for lon
                 in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords, coordsUFP)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.item()


def buildVolume(year, coords,
                path_=f"{pathDat}/GHSL/builtup_volume"):
    """
    Returns the buildup volume for the grid cell corresponding
    to the coordinates
    defined in the arguments of the function and the year.

    Args:
        coords: Coordinates [lat, lon]

    Returns:

    """
    from os import chdir, path
    from xarray import open_dataset
    from rioxarray import open_rasterio
    from math import floor, ceil
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
    # Read the dataset
    if not path.isfile(files[years.index(year)].replace(".tif", ".nc")):
        ds = open_rasterio(files[years.index(year)])
        # Reproject to EPSG4326 and save as a netcdf
        ds = ds.rio.reproject("EPSG:4326")
        ds = ds.sel(band=1).drop_vars("band").to_dataset(name="buildUp")
        ds = ds.rename({"x": "lon", "y": "lat"})
        # Save it to save time next time
        ds.to_netcdf(files[years.index(year)].replace(".tif", ".nc"),
                     encoding={"buildUp": {"dtype": "int32", "zlib": True}})
    ds = open_dataset(files[years.index(year)].replace(".tif", ".nc"))
    # Get the closest coordinate from the land use dataset
    ds_ = ds.sel(lat=slice(ceil(coords[0]+0.1), floor(coords[0]-0.1)),
                 lon=slice(floor(coords[1]-0.1), ceil(coords[1]+0.1)))
    coordsBuildUp = [[lat, lon] for lat in ds_.lat.values for lon
                     in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords, coordsBuildUp)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.buildUp.item()


def degreeUrban(year, coords,
                path_=f"{pathDat}/GHSL/degree_urbanisation"):
    """
    Returns the degree of urbanisation for the grid cell corresponding
    to the coordinates defined in the arguments of the function.

    Args:
        coords: coordinates to get the degree of urbanisation for

    Returns:
        Degree of urbanisation (float)
    """
    from os import chdir, path
    from xarray import open_dataset
    from rioxarray import open_rasterio
    from glob import glob1
    from numpy import argmin
    from math import floor, ceil
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
        ds = ds.sel(band=1).drop_vars("band").to_dataset(name="degUrb")
        if "spatial_ref" in ds.coords:
            ds = ds.drop_vars("spatial_ref")
        ds = ds.rename({"x": "lon", "y": "lat"})
        # Save it to save time next time
        ds.to_netcdf(files[years.index(year)].replace(".tif", ".nc"),
                     encoding={"degUrb": {"dtype": "int16", "zlib": True}})
    # Read the netcdf file
    ds = open_dataset(files[years.index(year)].replace(".tif", ".nc"))
    # Get the closest coordinate from the land use dataset
    ds_ = ds.sel(lat=slice(ceil(coords[0]+0.1), floor(coords[0]-0.1)),
                 lon=slice(floor(coords[1]-0.1), ceil(coords[1]+0.1)))
    coordsDegreeUrban = [[lat, lon] for lat in ds_.lat.values for lon
                         in ds_.lon.values]
    # Get the closest coordinate from the land use dataset
    coordClose = closest_node(coords, coordsDegreeUrban)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.degUrb.item()


def humanSettle(year, coords,
                path_=f"{pathDat}/GHSL/human_settlement_wsg84"):
    """
    Returns the Human Settlement data for the grid cell corresponding
    to the coordinates defined in the arguments of the function.

    Args:
        coords: coordinates to get the degree of urbanisation for

    Returns:
        Degree of urbanisation (float)
    """
    from os import chdir
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
    # Read the netcdf file
    ds = open_rasterio(files[years.index(year)])\
        .rename({"x": "lon", "y": "lat"})
    # Get the closest coordinate from the land use dataset
    ds_ = ds.sel(lat=coords[0], lon=coords[1], method="nearest")
    return ds_.item()


def getPM25(year, coords,
            path=f"{pathDat}/PM25"):
    from glob import glob1
    from pandas import DataFrame
    import os
    from math import ceil, floor
    from xarray import open_dataset

    os.chdir(path)
    # List the datasets
    dat = glob1(os.getcwd(), "*.nc")
    dat = DataFrame({
        "year": [int(x.split("-")[-1][:4]) for x in dat],
        "path": [f"{os.getcwd()}/{x}" for x in dat]
    }).sort_values(by="year").reset_index(drop=True)
    # Load the netcdf for the year into an xarray
    ds = open_dataset(dat.loc[dat.year == year].path.item())
    # Get the closest coordinate from the PM2.5 dataset
    ds_ = ds.sel(lat=slice(floor(coords[0]-0.1), ceil(coords[0]+0.1)),
                 lon=slice(floor(coords[1]-0.1), ceil(coords[1]+0.1)))
    coordsPM25 = [[lat, lon] for lat in ds_.lat.values for lon
                  in ds_.lon.values]
    # Get the closest coordinate from the PM2.5 dataset
    coordClose = closest_node(coords, coordsPM25)
    # Get the data for the corresponding grid cell
    ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
    return ds_.GWRPM25.item()


def readEmissions(year,
                  path=f"{pathDat}/CAMS_emissions/interpolated_"):
    from xarray import open_dataset
    from glob import glob1
    import pandas as pd
    import os

    def convDf(dat_):
        from pandas import read_parquet

        df = read_parquet(dat_)
        return df.drop(["lat_", "lon_", "spatial_ref"], axis=1)\
            .set_index(["lat", "lon"]).to_xarray()

    if not os.path.isfile(f"{path}/{year}.nc"):
        ds = convDf(f"{path}/{year}.parquet")
        ds.to_netcdf(f"{path}/{year}.nc",
                     encoding={"black-carbon": {"zlib": True,
                                                "complevel": 9},
                               "carbon-dioxide": {"zlib": True,
                                                  "complevel": 9},
                               "carbon-monoxide": {"zlib": True,
                                                   "complevel": 9},
                               "nitrogen-oxides": {"zlib": True,
                                                   "complevel": 9}})
        return ds

    # List the netcdf files
    filesEm = glob1(path, "*.nc")
    filesEm = pd.DataFrame({"year": [int(x.split(".")[0]) for x in filesEm],
                            "path": [f"{path}/{x}" for x in filesEm]})\
        .sort_values(by="year", ascending=True)\
        .reset_index(drop=True)
    return open_dataset(filesEm.loc[filesEm.year == year].path.item())


def getEmissions(coords, year):
    import pandas as pd
    # Subset for the specified coordinates etc. . .
    try:
        # Read the emissions dataset for the year
        ds = readEmissions(year)
        ds_ = ds.sel(lat=slice(coords[0]-0.02, coords[0]+0.02),
                     lon=slice(coords[1]-0.02, coords[1]+0.02))
        coords_ = [[lat, lon] for lat in ds_.lat.values for lon
                   in ds_.lon.values]
        # Get the closest coordinate from the land use dataset
        coordClose = closest_node(coords, coords_)
        # Get the data for the corresponding grid cell
        ds_ = ds_.sel(lat=coordClose[0], lon=coordClose[1])
        # Put the data to return in a list
        dat_ret = pd.DataFrame(
            {"blackCarbon": [ds_["black-carbon"].item()],
             "carbonDioxide": [ds_["carbon-dioxide"].item()],
             "carbonMonoxide": [ds_["carbon-monoxide"].item()],
             "nitrogenOxides": [ds_["nitrogen-oxides"].item()]})
    except Exception as e:
        print(e)
        dat_ret = pd.DataFrame({"blackCarbon": [None],
                                "carbonDioxide": [None],
                                "carbonMonoxide": [None],
                                "nitrogenOxides": [None]})
    return dat_ret


def gett2m(coords, year, path=f"{pathDat}/ERA5"):
    ds = xr.open_dataset(f"{path}/t2m_yearly.nc").rename({"latitude": "lat",
                                                          "longitude": "lon"})
    coords_ = coords.copy()
    if coords_[1] < 0:
        coords_[1] += 360
    try:
        # Get the closest coordinates from the NO2 dataset
        ds_ = ds.sel(lon=slice(coords_[1] - 0.2, coords_[1] + 0.2),
                     lat=slice(coords_[0] + 0.2, coords_[0] - 0.2),
                     time=f"{year}-12-31")
        coordst2m = [[lat, lon] for lat in ds_.lat.values for lon
                     in ds_.lon.values]
        # Get the closest coordinate from the NO2 dataset
        coordClose = closest_node(coords_, coordst2m)
        # Get the NO2 concentration for the coordClose coordinate
        ds_ = ds_.sel(lon=coordClose[1], lat=coordClose[0])
        return ds_.t2m.item()
    except Exception as e:
        print(coords, e)
        return None


def getStation_(coordsSt, year):
    """
    Constructs the dataset for a given year and statoin

    Args:
        year: Year to retrieve data for (int)
        coordsSt: dict which contains the coordinates for the stations

    Returns:
        df: Pandas dataframe
    """
    from pandas import DataFrame

    # Get the data
    try:
        df = getLand(year, coordsSt)
    except Exception:
        df = DataFrame({"year": [year]})
        for i in range(1, 8):
            df[f"land_{i}"] = None
    try:
        no2 = getNO2(year, coordsSt)
        df = df.assign(no2=no2.values.item())
    except Exception:
        df = df.assign(no2=None)
    try:
        df = df.assign(pop=getPop(year, coordsSt))
    except Exception:
        df = df.assign(pop=None)
    try:
        df = df.assign(buildUp=buildVolume(year, coordsSt))
    except Exception:
        df = df.assign(buildUp=None)
    try:
        df = df.assign(degreeUrb=degreeUrban(year, coordsSt))
    except Exception:
        df = df.assign(degreeUrb=None)
    try:
        df = df.assign(humanSettle=humanSettle(year, coordsSt))
    except Exception:
        df = df.assign(humanSettle=None)
    try:
        df = df.assign(pm25=getPM25(year, coordsSt))
    except Exception:
        df = df.assign(pm25=None)
    try:
        df = df.assign(t2m=gett2m(coordsSt, year))
    except Exception:
        df = df.assign(t2m=None)
    df = df.assign(lat=[coordsSt[0]], lon=[coordsSt[1]])
    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", 50)
    df = pd.read_csv(args.path_obs)

    # Directory to save intermediatary datasets
    path_training = f"{pathDat}/dat_"
    os.makedirs(path_training, exist_ok=True)

    df_dat = pd.concat([getStation_(dat[:2], int(dat[2])) for dat in
                        tqdm(df[["lat", "lon", "year"]].values)])
    df_dat = df_dat.reset_index(drop=True)
    # Emissions
    emissions = pd.concat([getEmissions(list(dat[:2]), int(dat[2])) for dat in
                           tqdm(df_dat[["lat", "lon", "year"]].values)])
    emissions = emissions.reset_index(drop=True)

    df_ = pd.concat([df[["lat", "lon", "year", "particle_number"]],
                     df_dat.drop(["year", "lat", "lon"],
                                 axis=1)],
                    axis=1)
    df_ = pd.concat([df_, emissions], axis=1)
    # And save it
    df_.dropna().loc[df_.year.isin(range(2000, 2021))]\
        .to_csv(f"{pathDat}/training_data.csv", index=False)
