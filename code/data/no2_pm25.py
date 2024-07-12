import os
import wget
import argparse
import shutil


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str,
                        help="Path to directory to download data.")
    args = parser.parse_args()

    # NO2
    urlNO2 = "https://figshare.com/ndownloader/articles/12968114/versions/4"
    os.makedirs(f"{args.path_data}/NO2_1km", exist_ok=True)
    os.chdir(f"{args.path_data}/NO2_1km")
    if not os.path.isfile(f"{args.path_data}/NO2_1km/12968114.zip"):
        wget.download(urlNO2)

    # Unpack it
    if not os.path.isfile(f"{args.path_data}/NO2_1km/1990_final_1km.tif"):
        shutil.unpack_archive(f"{args.path_data}/NO2_1km/12968114.zip")

    # No way to download the PM2.5 through a script
    print("Please download the PM2.5 from https://sites.wustl.edu/acag/datasets/surface-pm2-5/#V4.NA.03.")
    print(f"to {args.path_data}/PM2.5")