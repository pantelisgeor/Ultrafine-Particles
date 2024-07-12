import wget
import os
import json
import argparse
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str,
                        help="Path to directory to download data.")
    args = parser.parse_args()

    urls = [
        "https://zenodo.org/records/3939038/files/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif?download=1",
        "https://zenodo.org/records/3518026/files/PROBAV_LC100_global_v3.0.1_2016-conso_Discrete-Classification-map_EPSG-4326.tif?download=1",
        "https://zenodo.org/records/3518036/files/PROBAV_LC100_global_v3.0.1_2017-conso_Discrete-Classification-map_EPSG-4326.tif?download=1",
        "https://zenodo.org/records/3518038/files/PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif?download=1",
        "https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif?download=1"]

    os.makedirs(f"{args.path_data}/land_use", exist_ok=True)
    os.chdir(f"{args.path_data}/land_use")

    for url in urls:
        print(url.split('/')[-1])
        if os.path.isfile(f"{args.path_data}/land_use/{url.split('/')[-1]}"):
            continue
        wget.download(
            url,
            f"{args.path_data}/land_use/{url.split('/')[-1]}")

    # Copernicus land use classes
    copernicus_classes = {
        "0": "No input data available",
        "111": "Closed forest, evergreen needle leaf",
        "113": "Closed forest, deciduous needle forest",
        "112": "Closed forest, evergreen, broad leaf",
        "114": "Closed forest, deciduous broad leaf",
        "115": "Closed forest, mixed",
        "116": "Closed forest, unknown",
        "121": "Open forest, evergreen, needle leaf",
        "123": "Open forest, deciduous needle leaf",
        "122": "Open forest, evergreen broad leaf",
        "124": "Open forest, deciduous broad leaf",
        "125": "Open forest, mixed",
        "126": "Open forest, unknown",
        "20": "Shrubs",
        "30": "Herbaceous vegetation",
        "90": "Herbaceous wetland",
        "100": "Moss and lichen",
        "60": "Bare/sparse vegetation",
        "40": "Cultivated and managed vegetation/agriculture (cropland)",
        "50": "Urban/built up",
        "70": "Snow and Ice",
        "80": "Permanent water bodies",
        "200": "Open sea"
    }

    # Save the dictionary as a json file
    os.chdir(f"{args.path_data}/land_use")
    with open("copernicus_classes.json", "w", encoding="utf-8") as f:
        json.dump(copernicus_classes, f)

    # Land use mappings
    copernicus_mapping = {
        0: [0],   # No input data available
        1: [111, 112, 113, 114, 115, 116, 121,
            122, 123, 124, 125, 126],  # Forests
        2: [20, 30, 60, 100],  # Vegetation (low height)
        3: [80, 90],  # Inland water
        4: [40],  # Antrhopogenic driven vegetation (croplands)
        5: [50],  # Urban/built ip
        6: [70],  # Snow/Ice
        7: [200]  # Open sea
    }
    # Save the mappings
    with open("copernicus_mappings.json",
              "w") as f:
        json.dump(copernicus_mapping, f)
    # And pickle it
    with open("copernicus_mappings.pickle",
              "wb") as handle:
        pickle.dump(copernicus_mapping, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
