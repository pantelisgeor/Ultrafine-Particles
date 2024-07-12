import cdsapi
import argparse
import xarray as xr
import gc
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data",
                        type=str,
                        help="Path to directory to download data.")
    args = parser.parse_args()

    os.makedirs(f"{args.path_data}/ERA5", exist_ok=True)
    os.chdir(f"{args.path_data}/ERA5")

    c = cdsapi.Client()

    for year in range(2000, 2021, 1):
        print(f"\nDownloading t2m for year {year}.")
        if os.path.isfile(f"ERA5_{year}_2m_temperature.nc"):
            print(f"ERA5_{year}_2m_temperature.nc exists. . . Skipping")
            continue
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": "2m_temperature",
                    "year": str(year),
                    "month": [
                        "01", "02", "03", "04", "05", "06",
                        "07", "08", "09", "10", "11", "12",],
                    "day": [
                        "01", "02", "03", "04", "05", "06",
                        "07", "08", "09", "10", "11", "12",
                        "13", "14", "15", "16", "17", "18",
                        "19", "20", "21", "22", "23", "24",
                        "25", "26", "27", "28", "29", "30", "31", ],
                    "time": [
                        "00:00", "01:00", "02:00",
                        "03:00", "04:00", "05:00",
                        "06:00", "07:00", "08:00",
                        "09:00", "10:00", "11:00",
                        "12:00", "13:00", "14:00",
                        "15:00", "16:00", "17:00",
                        "18:00", "19:00", "20:00",
                        "21:00", "22:00", "23:00",],
                },
                f"ERA5_{year}_2m_temperature.nc")
        except Exception as e:
            print(e)
            continue

    if not os.path.isfile("t2m_yearly.nc"):
        print("Creating yearly averages dataset")
        # Loop through them and calculate the yearly averages
        for c, year in enumerate(range(2000, 2021, 1)):
            print(year)
            # Read the netcdf into an xarray
            ds_ = xr.open_dataset(f"ERA5_{year}_2m_temperature.nc")
            if c == 0:
                ds = ds_.resample(time="1Y").mean()
            else:
                ds = ds.merge(ds_.resample(time="Y").mean())
            del ds_, c, year
            gc.collect()

        # Save it
        ds.to_netcdf("t2m_yearly.nc",
                     encoding={"t2m": {"zlib": True,
                                       "complevel": 6}})
