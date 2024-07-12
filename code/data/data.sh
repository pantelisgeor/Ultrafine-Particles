#!/bin/bash

path_data=$1

# Downloads the ERA5 data needed
python ERA5.py --path_data=$path_data

# Download the land use datasets
python copernicus_land.py --path_data=$path_data

# Download the NO2 datasets
python no2_pm25.py --path_data=$path_data

# Global Human Layer datasets
python GHLS.py --path_data=$path_data

# CAMS emissions inventory
python CAMS_emissions.py --path_data=$path_data
# And interpolate it wrt GHLS
for year in $(seq 2000 2020 1)
do
    python CAMS_emissions_interp.py --path_data=$path_data --year=$year
done

# Finally use the data to construct the training set
python create_training_set.py --path_data=$path_data --path_obs=UFP_data.csv