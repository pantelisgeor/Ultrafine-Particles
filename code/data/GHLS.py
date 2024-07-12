import os
import wget
import shutil
import glob
import rioxarray as rxr
import gc
import re
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str,
                    help="Path to directory to download data.")
args = parser.parse_args()

os.makedirs(f"{args.path_data}/GHSL", exist_ok=True)

# ====================== HUMAN SETTLEMENTS ===================== #
path = f"{args.path_data}/GHSL/human_settlement_wsg84"
os.makedirs(path, exist_ok=True)

files = [
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2000_GLOBE_R2023A_4326_3ss/V1-0/GHS_POP_E2000_GLOBE_R2023A_4326_3ss_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2005_GLOBE_R2023A_4326_3ss/V1-0/GHS_POP_E2005_GLOBE_R2023A_4326_3ss_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2010_GLOBE_R2023A_4326_3ss/V1-0/GHS_POP_E2010_GLOBE_R2023A_4326_3ss_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2015_GLOBE_R2023A_4326_3ss/V1-0/GHS_POP_E2015_GLOBE_R2023A_4326_3ss_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.zip"
]

for f in files:
    print(f.split("/")[-1])
    wget.download(f, f"{path}/{f.split('/')[-1]}")

# Unzip them
os.chdir(path)
[shutil.unpack_archive(x) for x in glob.glob1(path, "*.zip")]

# =================== DEGREE OF URBANISATION ===================== #

path = f"{args.path_data}/GHSL/degree_urbanisation"
os.makedirs(path, exist_ok=True)

files = [
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2025_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2015_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2015_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2010_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2010_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2005_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2005_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2000_GLOBE_R2023A_54009_1000/V1-0/GHS_SMOD_E2000_GLOBE_R2023A_54009_1000_V1_0.zip"
]

for f in files:
    if os.path.isfile(f"{path}/{f.split('/')[-1]}"):
        continue
    print(f.split("/")[-1])
    wget.download(f, f"{path}/{f.split('/')[-1]}")

# Unzip them
os.chdir(path)
[shutil.unpack_archive(x) for x in glob.glob1(path, "*.zip")]

# List the .tif files
tifs = sorted(glob.glob1(path, "*.tif"))

for t in tqdm(tifs):
    # Get the year from the filename
    year = int(re.search(r"E(\d{4})", t).group(1))
    # Construct the date
    date = pd.to_datetime([f"01-01-{year}"])[0]
    # Read the dataset
    ds_temp = rxr.open_rasterio(t)
    # Reproject to WSG84
    ds_temp = ds_temp.rio.reproject("EPSG:4326")
    gc.collect()
    # Add the date
    ds_temp = ds_temp\
        .to_dataset(name="urb_degree")\
        .expand_dims(time=[date])
    if t == tifs[0]:
        ds = ds_temp
    else:
        ds = ds.merge(ds_temp)
    del ds_temp
    gc.collect()


# =================== Builtup volume ===================== #
path = f"{args.path_data}/GHSL/builtup_volume"
os.makedirs(path, exist_ok=True)

files = [
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2025_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_V_E2025_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_V_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2015_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_V_E2015_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2010_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_V_E2010_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2005_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_V_E2005_GLOBE_R2023A_54009_1000_V1_0.zip",
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/GHS_BUILT_V_E2000_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_V_E2000_GLOBE_R2023A_54009_1000_V1_0.zip"
]

for f in files:
    if os.path.isfile(f"{path}/{f.split('/')[-1]}"):
        continue
    print(f.split("/")[-1])
    wget.download(f, f"{path}/{f.split('/')[-1]}")

# Unzip them
os.chdir(path)
[shutil.unpack_archive(x) for x in glob.glob1(path, "*.zip")]

# List the .tif files
tifs = sorted(glob.glob1(path, "*.tif"))

for t in tqdm(tifs):
    # Get the year from the filename
    year = int(re.search(r"E(\d{4})", t).group(1))
    # Construct the date
    date = pd.to_datetime([f"01-01-{year}"])[0]
    # Read the dataset
    ds_temp = rxr.open_rasterio(t)
    # Reproject to WSG84
    ds_temp = ds_temp.rio.reproject("EPSG:4326")
    gc.collect()
    # Add the date
    ds_temp = ds_temp\
        .to_dataset(name="urb_degree")\
        .expand_dims(time=[date])
    if t == tifs[0]:
        ds = ds_temp
    else:
        ds = ds.merge(ds_temp)
    del ds_temp
    gc.collect()
