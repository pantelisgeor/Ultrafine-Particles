import os
from glob import glob
import cdsapi
import argparse
from tqdm import tqdm
from glob import glob1
import pandas as pd
import xarray as xr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str,
                        help="Path to directory to download data.")
    args = parser.parse_args()

    print("\n\nMake sure you have set up your Atmosphere Data Store account")
    print("at https://ads.atmosphere.copernicus.eu/cdsapp#!/home\n\n")

    # Function definitions
    def untar(filename,
              path=f"{args.path_data}/CAMS_emissions/anthropogenic"):
        print(filename)
        os.system(f"cd {path} && tar -xvf {filename}")

    def unzip(filename,
              path=f"{args.path_data}/CAMS_emissions/anthropogenic"):
        print(filename)
        os.system(f"cd {path} && unzip {filename}")

    def allEmissions(em_):
        for yr in tqdm(dat.loc[dat.em == em_].year.values):
            if yr == dat.loc[dat.em == em_].year.values[0]:
                ds = xr.open_dataset(dat.loc[(dat.em == em_) &
                                             (dat.year == yr)].path.item(),
                                     decode_times=False)
                ds = ds["sum"].mean(dim="time")\
                    .to_dataset(name=em_)\
                    .expand_dims(time=pd.to_datetime([f"{yr}-12-31"]))
            else:
                ds_ = xr.open_dataset(dat.loc[(dat.em == em_) &
                                              (dat.year == yr)].path.item(),
                                      decode_times=False)
                ds_ = ds_["sum"].mean(dim="time")\
                    .to_dataset(name=em_)\
                    .expand_dims(time=pd.to_datetime([f"{yr}-12-31"]))
                ds = ds.merge(ds_)
                del ds_
            del yr
        return ds

    os.makedirs(f"{args.path_data}/CAMS_emissions",
                exist_ok=True)
    os.makedirs(f"{args.path_data}/CAMS_emissions/anthropogenic",
                exist_ok=True)

    os.chdir(f"{args.path_data}/CAMS_emissions/anthropogenic")

    # Initiate the API
    c = cdsapi.Client()

    for year in range(2000, 2021):
        if os.path.isfile(f'{year}.zip'):
            continue
        c.retrieve(
            'cams-global-emission-inventories',
            {
                'version': 'latest',
                'format': 'zip',
                'source': 'anthropogenic',
                'variable': [
                    'black_carbon', 'carbon_dioxide',
                    'carbon_monoxide', 'nitrogen_oxides'],
                'year': str(year),
            },
            f'{year}.zip')

    # List the downloaded zip files
    files = glob("*.zip")

    # Unzip them all
    for x in tqdm(files):
        if os.path.isfile(
                f"CAMS-GLOB-ANT_v4.2_carbon-dioxide_{x.split('.')[0]}.nc"):
            continue
        unzip(x)

    # List all the netcdf files in the path_dat directory
    dat = glob1(f"{args.path_data}/CAMS_emissions/anthropogenic",
                "CAMS-GLOB-*.nc")
    dat = pd.DataFrame({
        "em": [x.split("_")[-2] for x in dat],
        "year": [int(x.split("_")[-1].split(".")[0]) for x in dat],
        "path": [f"{args.path_data}/CAMS_emissions/anthropogenic/{x}"
                 for x in dat]})\
        .sort_values(by=["em", "year"]).reset_index(drop=True)

    # For a given emission source, load the dataset for each year,
    # calculate the mean of the sum variable (sum of all the emission sources)
    # and combine them
    if not os.path.isfile("all_emissions.nc"):
        for c, em_ in enumerate(dat.em.unique()):
            print(c)
            if c == 0:
                print("1st...", em_)
                ds = allEmissions(em_=em_)
            else:
                print(em_)
                ds = ds.merge(allEmissions(em_=em_))
        # Save it
        ds.to_netcdf("all_emissions.nc",
                     encoding={var: {"zlib": True,
                                     "complevel": 9} for var in ds.data_vars})
