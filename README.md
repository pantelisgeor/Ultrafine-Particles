# Global high-resolution ultrafine particle number concentrations through data fusion with machine learning
Pantelis Georgiades, Matthias Kohl, Mihalis A. Nicolaou, Theodoros Christoudias, Andrea Pozzer, Constantine Dovrolis, Jos Lelieveld

**Correspondence:** Pantelis Georgiades (p.georgiades@cyi.ac.cy) and Jos Lelieveld (jos.lelieveld@mpic.de)

This repository contains the data retrieval scripts and source code to produce the results presented in:

## PAPER CITATION


## Abstract

Atmospheric pollution causes millions of excess deaths annually, with particulate matter (PM) being a major concern. While research has traditionally focused on PM<sub>10</sub> and PM<sub>2.5</sub>, ultrafine particles (UFPs, diameter~<~100~nm) have emerged as a critical human health risk due to their ability to penetrate deeply into the respiratory system, transmigrate into the bloodstream and induce systemic health impacts. The total particle number concentration (PNC) serves as a proxy measure for UFP prevalence, as UFPs dominate particle number counts despite contributing minimally to total particle mass. This study presents the first global datasets of PNCs and UFPs at 1 km resolution over land by combining ground station measurements with machine learning. We developed an XGBoost model to predict annual PNC levels from 2010-2019, integrating diverse environmental and anthropogenic variables available at the global scale.

## Instructions:

You can download a copy of all the files in this repository by cloning the git repository:

```
git clone https://github.com/pantelisgeor/Ultrafine-Particles
```

**Note** The code provided was developed and tested on a node equipped with 256 GB of RAM and 2 AMD EPYC Milan 64 core CPUs, running Linux.

### Setting up your environment

You'll need a working Python 3 environment with the following libraries:
1. Standard Scientific libraries: numpy, pandas, scipy, matplotlib, cartopy, scikit-learn, pyarrow.
2. MAPIE library (https://mapie.readthedocs.io/)
3. SHAP library (https://shap.readthedocs.io/)
4. Spatial data: xarray, netcdf4, rasterio
5. Other libraries: tqdm, python-wget, cdsapi


#### Data Retrieval

The code is written exclusively in Python and uses a number of bash scripts to execute the workload. First, to retrieve the data needed *data.sh* is called. It takes one argument, the path to the directory where the data are to be stored. (The Copernicus Data Store and Copernicus Atmospheric Data Store APIs are utilised, you can register and set up the APIs at cds.climate.copernicus.eu and ads.atmosphere.copernicus.eu, respectively).

To execute the bash script, run the following commands:

```
cd code
chmod +x data.sh
./data.sh ~/Data
```
**Note** The bash script downloads and processes hundreds of GBs of data. Make sure you have the appropriate compute and storage capabilities!

#### Machine Learning

To construct the inference data run the **make_data.sh** bash script in `code/ML`. The bash script takes two positional arguments, first the path to the directory where the retrieved datasets are stored and second the path to the directory where the inference data are to be stored. To run the bash script:

```
cd code/ML
chmod +x make_data.sh
./make_data.sh ~/data ~/data_inference
```

To train an XGBoost regression model and perform inference the **train_predict.py** bash script is provided. The train_predict.py script takes four positional arguments:
1. Path to the training dataset created.
2. Path to the directory where the trained model will be stored.
3. Path to the MinMaxScaler for the feature set (provided in `code\ML\scaler_feats.joblib`).
4. Path to the directory where inference data is stored.

```
cd code/ML
chmod +x train_predict.sh
./train_predict.py ~/Data ~/trained_models <path to scaler_feats.joblib> ~/data_inference
```

## License

All Python source code is made available under the MIT license. You can freely use and modify the code, without warranty, so long as you provide attributions to the authors. See 'LICENSE-MIT.txt' for the full license text.

The manuscript text, figures and data/models produced as part of this research are available under the [Creative Commons Attribution 4.0 License (CC-BY)][cc-by]. See `LICENSE-CC-BY.txt` for the full license text.

[cc-by]: https://creativecommons.org/licenses/by/4.0/
