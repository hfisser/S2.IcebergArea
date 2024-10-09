# S2.IcebergArea
Detect icebergs and sea ice floes in Sentinel-2 data, delineate their above-waterline area, and distinguish icebergs from sea ice.

### Citation
Publication..

### Installation
Install dependencies and the S1.IcebergArea package in an Anaconda environment:

```shell
conda env create -f environment.yml  # conda environment, install dependencies needed for Sentinel1Denoised
conda activate s1icebergarea  # activate it
pip install https://github.com/nansencenter/sentinel1denoised/archive/v1.4.0.tar.gz  # install Sentinel1Denoised package for noise correction in HV channel
pip install ...  # install S1.IcebergArea package with its dependencies
```
### Example
```python
import os
import geopandas as gpd
from S2IcebergArea.S2IcebergArea import S2IcebergArea

dir_safe = "/my/s1_data/........"  # unzipped
aoi = gpd.read_file("/my/aois/aoi.gpkg")
s2_iceberg_area = S2IcebergArea()  # initialize S2IcebergArea class
s2_iceberg_area.prepare_s2(dir_safe, os.path.dirname(dir_safe))  # run calibration, noise removal
icebergs = s2_iceberg_area.run_model(aoi=aoi)  # run area model
```
### Data
Sentinel-2 level 1C data.

### Output
A geopandas GeoDataFrame. The GeoDataFrame contains the delineated iceberg and sea ice outlines, reflectance statistics, and the above-waterline area.

### Algorithm

### Background
