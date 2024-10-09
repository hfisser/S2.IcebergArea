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
    description = ("Package detecting icebergs in Sentinel-2, delineate their area, and distinguish them from sea ice floes"),
    license = "Tbd",
    long_description=read("README.md"),
    install_requires = [
        "geopandas",
        "numpy",
        "rasterio",
        "rasterstats",
        "scikit-learn"
    ],
    packages = find_packages(where="src"),
    package_dir = {"": "src"},
    package_data = {"": ["*.pickle"]},
    entry_points = {
        "console_scripts": [
        ]
    },
    include_package_data=True,
)
