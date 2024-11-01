import os
from setuptools import setup, find_packages

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setup(
    name = "S2.IcebergArea",
    version = "0.0.0",
    author = "Henrik Fisser",
    author_email = "henrik.fisser@uit.no",
    description = ("Package detecting icebergs in Sentinel-2, delineate their area, and distinguish icebergs and sea ice floes"),
    license = "Tbd",
    long_description=read("README.md"),
    install_requires = [
        "s2cloudless",
        #"fiona==1.8.22",
        #"geopandas==0.12.2",
        "numpy==1.23.5",
        "rasterio==1.2.10",
        "rasterstats==0.20.0",
        "scikit-image==0.19.3",
        "scikit-learn==1.2.0",
        "scipy==1.10.0"
    ],
    packages = find_packages(where="src"),
    package_dir = {"": "src"},
    package_data = {"": ["*.pickle"]},
    entry_points = {
        "console_scripts": []
    },
    include_package_data=True,
)
