# Mapping Atmospheric Ultrafine Particles from the Global to the Local Scale
Pantelis Georgiades, Matthias Kohl, Mihalis A. Nicolaou, Theodoros Christoudias, Andrea Pozzer, Constantine Dovrolis, Jos Lelieveld

**Correspondence:** Pantelis Georgiades (p.georgiades@cyi.ac.cy) and Jos Lelieveld (jos.lelieveld@mpic.de)

This repository contains the data and source code to produce the results presented in:

# ADD PAPER CITATION

## Abstract

Atmospheric pollution is a major concern due to its well-documented and detrimental impacts on human health, with millions of excess deaths attributed to it annually. Particulate matter (PM), comprising airborne pollutants in the form of solid and liquid particles suspended in the air, has been particularly concerning. Historically, research has focused on PM with an aerodynamic diameter less than 10~$\mu$m (PM$_{10}$) and 2.5~$\mu$m (PM$_{2.5}$), referred to as coarse and fine particulate matter, respectively. The long term exposure to both classes of PM have been shown to impact human health, being linked to a range of respiratory and cardiovascular complications. Recently, attention has been drawn to the lower end of the size distribution, specifically \textit{ultrafine particles} (UFPs), with an aerodynamic diameter less than  100~nm (PM$_{0.1}$). UFPs can deeply infiltrate the respiratory system, reach the bloodstream, and have been increasingly associated with chronic health conditions, including cardiovascular disease. Accurate mapping of UFP concentrations at high spatial resolution is crucial considering strong gradients near the sources. However, due to the relatively recent focus on this class of PM, there is a scarcity of long-term measurements, particularly on the global scale. In this study, we employed a machine learning methodology to produce the first global maps of UFP concentrations at high spatial resolution (1~km) by leveraging limited ground station measureWments worldwide. We trained an XGBoost model to predict annual UFP concentrations for a decade (2010-2019) and utilized the conformal prediction framework to provide reliable prediction intervals. This approach not only fills the current data gaps of global high-resolution UFP concentrations to enable comprehensive, data-informed assessments of the health implications associated with UFP exposure.

## Instructions:

You can download a copy of all the files in this repository by cloning the git repository:

```
git clone https://github.com/pantelisgeor/Ultrafine-Particles
```

**Note** The code provided was developed and tested on a node equipped node with 256 GB of RAM and 2 AMD EPYC Milan 64 core CPUs, running Linux.

The code is written exclusively in Python and uses a number of bash scripts to execute the workload. First, to retrieve the data needed *data.sh* is called. It takes one argument, the path to the directory where the data are to be stored. (The Copernicus Data Store and Copernicus Atmospheric Data Store APIs are utilised, you can register and set up the APIs at cds.climate.copernicus.eu and ads.atmosphere.copernicus.eu, respectively).

To execute the bash script, run the following commands:

```
chmod +x data.sh
./data.sh ~/Data
```
**Note** The bash script downloads and processes hundreds of GBs of data. Make sure you have the appropriate compute and storage capabilities!




## License

All Python source code is made available under the MIT license. You can freely use and modify the code, without warranty, so long as you provide attributions to the authors. See 'LICENSE-MIT.txt' for the full license text.

The manuscript text, figures and data/models produced as part of this research are available under the [Creative Commons Attribution 4.0 License (CC-BY)][cc-by]. See `LICENSE-CC-BY.txt` for the full license text.

[cc-by]: https://creativecommons.org/licenses/by/4.0/